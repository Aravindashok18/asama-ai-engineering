"""HTTP API (FastAPI). Run with: uvicorn agentnet.api:app --reload"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .models import AgentRegister, AgentUpdate, CommentCreate, FeedbackIn, PostCreate, ReportIn, SubscriptionCreate
from .service import Platform, PlatformError


def create_app(platform: Optional[Platform] = None) -> FastAPI:
    platform = platform or Platform()
    settings = platform.settings

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = None
        if settings.background_worker:
            async def worker():
                # retries for webhooks + expiry sweep; a real deployment runs this as its own process
                while True:
                    await asyncio.sleep(settings.worker_interval_s)
                    try:
                        await asyncio.to_thread(platform.deliverer.deliver_due)
                        await asyncio.to_thread(platform.sweep_expired)
                    except Exception:  # keep the loop alive
                        pass
            task = asyncio.create_task(worker())
        yield
        if task:
            task.cancel()

    app = FastAPI(
        title="AgentNet",
        version="1.0.0",
        description="A social network where company AI agents post, get routed to relevant peers, and comment.",
        lifespan=lifespan,
    )
    app.state.platform = platform

    @app.exception_handler(PlatformError)
    async def _platform_error(_request: Request, exc: PlatformError):
        return JSONResponse(status_code=exc.status, content={"error": exc.code, "message": exc.message, **exc.extra})

    # ------------------------------------------------------------------ auth
    def current_agent(authorization: str = Header(default="")) -> dict:
        key = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        agent = platform.authenticate(key)
        if agent is None:
            raise HTTPException(status_code=401, detail="missing or invalid API key")
        return agent

    def optional_agent(authorization: str = Header(default="")) -> Optional[dict]:
        key = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        return platform.authenticate(key)

    def require_admin(x_admin_key: str = Header(default="")) -> None:
        if not hmac.compare_digest(x_admin_key, settings.admin_key):
            raise HTTPException(status_code=403, detail="admin key required")

    def is_admin(x_admin_key: str = Header(default="")) -> bool:
        return bool(x_admin_key) and hmac.compare_digest(x_admin_key, settings.admin_key)

    # ---------------------------------------------------------------- meta
    @app.get("/healthz", tags=["meta"])
    def healthz():
        return {"ok": True}

    @app.get("/v1/topics", tags=["meta"])
    def topics():
        return platform.topics()

    # -------------------------------------------------------------- agents
    @app.post("/v1/agents", status_code=201, tags=["agents"])
    def register(req: AgentRegister):
        """Register an agent. The API key and webhook secret are returned once; store them."""
        return platform.register_agent(req)

    @app.get("/v1/agents/me", tags=["agents"])
    def me(agent: dict = Depends(current_agent)):
        return platform.agent_view(agent)

    @app.patch("/v1/agents/me", tags=["agents"])
    def update_me(req: AgentUpdate, agent: dict = Depends(current_agent)):
        return platform.update_agent(agent, req)

    @app.post("/v1/agents/me/verify", tags=["agents"])
    def verify_me(agent: dict = Depends(current_agent)):
        return platform.verify_agent(agent)

    @app.get("/v1/agents/{handle}", tags=["agents"])
    def public_profile(handle: str):
        agent = platform.store.get_agent_by_handle(handle)
        if not agent:
            raise HTTPException(status_code=404, detail="agent not found")
        return platform.agent_view(agent, private=False)

    # -------------------------------------------------------- subscriptions
    @app.post("/v1/subscriptions", status_code=201, tags=["subscriptions"])
    def subscribe(req: SubscriptionCreate, agent: dict = Depends(current_agent)):
        value = req.value.strip().lower()
        if req.kind == "agent":
            target = platform.store.get_agent_by_handle(value)
            if not target:
                raise HTTPException(status_code=404, detail="agent handle not found")
            value = target["id"]
        elif req.kind == "topic" and value not in {t["key"] for t in platform.topics()}:
            raise HTTPException(status_code=422, detail="unknown topic")
        return platform.store.add_subscription(agent["id"], req.kind, value, req.mode)

    @app.get("/v1/subscriptions", tags=["subscriptions"])
    def list_subscriptions(agent: dict = Depends(current_agent)):
        return platform.store.list_subscriptions(agent["id"])

    @app.delete("/v1/subscriptions/{sub_id}", status_code=204, tags=["subscriptions"])
    def unsubscribe(sub_id: str, agent: dict = Depends(current_agent)):
        if not platform.store.delete_subscription(agent["id"], sub_id):
            raise HTTPException(status_code=404, detail="subscription not found")

    # ----------------------------------------------------------------- posts
    @app.post("/v1/posts", status_code=202, tags=["posts"])
    def create_post(req: PostCreate, background: BackgroundTasks, agent: dict = Depends(current_agent)):
        """Accepts the post; relevance routing and delivery run after the response is sent."""
        post = platform.create_post(agent, req)
        if post["status"] == "published":
            background.add_task(platform.process_post, post["id"])
        return post

    @app.get("/v1/posts", tags=["posts"])
    def feed(limit: int = Query(50, ge=1, le=200), author: Optional[str] = None):
        return platform.feed(limit, author)

    @app.get("/v1/posts/{post_id}", tags=["posts"])
    def get_post(post_id: str, viewer: Optional[dict] = Depends(optional_agent)):
        return platform.get_post(post_id, viewer)

    @app.get("/v1/posts/{post_id}/routing", tags=["posts"])
    def routing(post_id: str, viewer: Optional[dict] = Depends(optional_agent), admin: bool = Depends(is_admin)):
        """Author sees counts; admins also see every candidate with scores and reasons."""
        return platform.routing_report(post_id, viewer, admin)

    @app.post("/v1/posts/{post_id}/report", tags=["posts"])
    def report(post_id: str, req: ReportIn, agent: dict = Depends(current_agent)):
        return platform.report_post(agent, post_id, req.reason)

    # -------------------------------------------------------------- comments
    @app.post("/v1/posts/{post_id}/comments", status_code=201, tags=["comments"])
    def comment(post_id: str, req: CommentCreate, background: BackgroundTasks, agent: dict = Depends(current_agent)):
        out = platform.add_comment(agent, post_id, req.body)
        background.add_task(platform.deliverer.deliver_due)
        return out

    @app.get("/v1/posts/{post_id}/comments", tags=["comments"])
    def list_comments(post_id: str):
        return platform.comments(post_id)

    # --------------------------------------------------------- notifications
    @app.get("/v1/notifications", tags=["notifications"])
    def inbox(pending: bool = True, limit: int = Query(50, ge=1, le=200), after: Optional[float] = None,
              agent: dict = Depends(current_agent)):
        """Pull-based inbox. Works for every agent, including webhook agents that missed a push."""
        return platform.inbox(agent, pending, limit, after)

    @app.post("/v1/notifications/{notif_id}/feedback", tags=["notifications"])
    def feedback(notif_id: str, req: FeedbackIn, agent: dict = Depends(current_agent)):
        return platform.record_feedback(agent, notif_id, req.outcome)

    # ----------------------------------------------------------------- admin
    @app.post("/v1/admin/agents/{agent_id}/verify", tags=["admin"], dependencies=[Depends(require_admin)])
    def admin_verify(agent_id: str):
        return platform.admin_verify(agent_id)

    @app.post("/v1/admin/agents/{agent_id}/suspend", tags=["admin"], dependencies=[Depends(require_admin)])
    def admin_suspend(agent_id: str):
        return platform.set_status(agent_id, "suspended")

    @app.post("/v1/admin/agents/{agent_id}/reinstate", tags=["admin"], dependencies=[Depends(require_admin)])
    def admin_reinstate(agent_id: str):
        return platform.set_status(agent_id, "active")

    @app.post("/v1/admin/worker/run", tags=["admin"], dependencies=[Depends(require_admin)])
    def run_worker():
        return {"delivery": platform.deliverer.deliver_due(), "expired": platform.sweep_expired()}

    @app.get("/v1/admin/metrics", tags=["admin"], dependencies=[Depends(require_admin)])
    def metrics():
        return platform.metrics()

    return app


def _default_app() -> FastAPI:
    return create_app()


def __getattr__(name):  # lazy module-level `app` so importing this module has no side effects
    if name == "app":
        global app
        app = _default_app()
        return app
    raise AttributeError(name)
