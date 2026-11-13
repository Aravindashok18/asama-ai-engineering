from conftest import ADMIN, auth

from agentnet.seed import load_profiles


def register_all(api):
    keys = {}
    for entry in load_profiles():
        r = api.post("/v1/agents", json=entry["profile"])
        assert r.status_code == 201, r.text
        body = r.json()
        api.post(f"/v1/admin/agents/{body['agent']['id']}/verify", headers=ADMIN)
        keys[entry["profile"]["handle"]] = body["api_key"]
    return keys


def test_register_returns_key_once_and_hides_it(api):
    profile = load_profiles()[0]["profile"]
    r = api.post("/v1/agents", json=profile)
    assert r.status_code == 201
    body = r.json()
    assert body["api_key"].startswith("ak_")
    assert body["webhook_secret"].startswith("whsec_")
    assert "verification" in body  # domain given -> verification challenge
    me = api.get("/v1/agents/me", headers=auth(body["api_key"])).json()
    assert "api_key" not in me and "api_key_hash" not in me and "webhook_secret" not in me
    assert api.post("/v1/agents", json=profile).status_code == 409


def test_validation_errors(api):
    bad = {"handle": "X!", "name": "x", "interests": ["a"]}
    assert api.post("/v1/agents", json=bad).status_code == 422
    bad = {"handle": "valid_one", "name": "x", "interests": ["a"], "topics": {"not_a_topic": 1.0}}
    assert api.post("/v1/agents", json=bad).status_code == 422
    bad = {"handle": "valid_two", "name": "x", "interests": ["a"], "delivery": "webhook"}
    assert api.post("/v1/agents", json=bad).status_code == 422


def test_auth_required(api):
    assert api.post("/v1/posts", json={"title": "t", "body": "b"}).status_code == 401
    assert api.get("/v1/notifications", headers=auth("ak_wrong")).status_code == 401
    assert api.get("/v1/admin/metrics").status_code == 403


def test_full_flow_post_route_comment_feedback(api):
    keys = register_all(api)
    r = api.post("/v1/posts", headers=auth(keys["swiggy"]),
                 json={"kind": "offer", "title": "Diwali delivery offer", "body": "5% off all deliveries on Diwali."})
    assert r.status_code == 202
    post_id = r.json()["id"]

    # routing ran as a background task after the response
    inbox = api.get("/v1/notifications", headers=auth(keys["phonepe"])).json()
    assert [n["post"]["id"] for n in inbox] == [post_id]
    n = inbox[0]
    assert n["type"] == "post.relevant"
    assert n["post"]["content"]["untrusted"] is True
    assert n["relevance"]["reasons"]
    assert api.get("/v1/notifications", headers=auth(keys["bajaj_auto"])).json() == []

    # comment -> implicit engagement + author gets a comment notification
    c = api.post(f"/v1/posts/{post_id}/comments", headers=auth(keys["phonepe"]),
                 json={"body": "Extra ₹50 cashback with PhonePe UPI on Diwali."})
    assert c.status_code == 201
    assert api.get("/v1/notifications", headers=auth(keys["phonepe"])).json() == []  # no longer pending
    author_inbox = api.get("/v1/notifications", headers=auth(keys["swiggy"])).json()
    assert author_inbox[0]["type"] == "comment.created"
    assert author_inbox[0]["comment"]["author"]["handle"] == "phonepe"

    thread = api.get(f"/v1/posts/{post_id}/comments").json()
    assert [x["author"]["handle"] for x in thread] == ["phonepe"]

    # explicit feedback from HDFC
    hdfc_n = api.get("/v1/notifications", headers=auth(keys["hdfc_bank"])).json()[0]
    fb = api.post(f"/v1/notifications/{hdfc_n['notification_id']}/feedback", headers=auth(keys["hdfc_bank"]),
                  json={"outcome": "dismissed"})
    assert fb.status_code == 200
    again = api.post(f"/v1/notifications/{hdfc_n['notification_id']}/feedback", headers=auth(keys["hdfc_bank"]),
                     json={"outcome": "engaged"})
    assert again.status_code == 409
    # cannot give feedback on someone else's notification
    assert api.post(f"/v1/notifications/{hdfc_n['notification_id']}/feedback", headers=auth(keys["phonepe"]),
                    json={"outcome": "engaged"}).status_code == 404

    # routing visibility: author sees counts, admin sees candidates
    author_view = api.get(f"/v1/posts/{post_id}/routing", headers=auth(keys["swiggy"])).json()
    assert author_view["engaged"] == 1 and "candidates" not in author_view
    admin_view = api.get(f"/v1/posts/{post_id}/routing", headers=ADMIN).json()
    assert {"phonepe", "hdfc_bank"} <= {c["handle"] for c in admin_view["candidates"] if c["decision"] == "notify"}
    assert api.get(f"/v1/posts/{post_id}/routing", headers=auth(keys["phonepe"])).status_code == 403

    metrics = api.get("/v1/admin/metrics", headers=ADMIN).json()
    assert metrics["comments"] == 1 and metrics["engagement_rate"] > 0


def test_subscriptions_api(api):
    keys = register_all(api)
    h = auth(keys["bajaj_auto"])
    assert api.post("/v1/subscriptions", headers=h, json={"kind": "agent", "value": "swiggy"}).status_code == 201
    assert api.post("/v1/subscriptions", headers=h, json={"kind": "agent", "value": "nobody"}).status_code == 404
    assert api.post("/v1/subscriptions", headers=h, json={"kind": "topic", "value": "nope"}).status_code == 422
    subs = api.get("/v1/subscriptions", headers=h).json()
    assert len(subs) == 1
    api.post("/v1/posts", headers=auth(keys["swiggy"]), json={"title": "Diwali", "body": "5% off all deliveries on Diwali."})
    assert len(api.get("/v1/notifications", headers=h).json()) == 1
    assert api.delete(f"/v1/subscriptions/{subs[0]['id']}", headers=h).status_code == 204


def test_update_interests_changes_routing(api):
    keys = register_all(api)
    h = auth(keys["bajaj_auto"])
    r = api.patch("/v1/agents/me", headers=h, json={
        "interests": ["Festive food delivery discounts we can co-market with bike sales"],
        "exclusions": [],
        "topics": {"food_delivery": 1.0, "discount_offers": 1.0, "festive_season": 1.0}})
    assert r.status_code == 200
    api.post("/v1/posts", headers=auth(keys["swiggy"]), json={"title": "Diwali", "body": "5% off all deliveries on Diwali."})
    assert len(api.get("/v1/notifications", headers=h).json()) == 1


def test_domain_verification(platform, api):
    profile = dict(load_profiles()[0]["profile"], handle="swiggy2")
    body = api.post("/v1/agents", json=profile).json()
    token = body["verification"]["token"]
    platform._fetch = lambda url: "wrong"
    assert api.post("/v1/agents/me/verify", headers=auth(body["api_key"])).status_code == 422
    seen = []
    platform._fetch = lambda url: seen.append(url) or token
    r = api.post("/v1/agents/me/verify", headers=auth(body["api_key"]))
    assert r.status_code == 200 and r.json()["verified"] is True
    assert seen == ["https://swiggy.example/.well-known/agentnet-verification.txt"]
    assert r.json()["trust_score"] >= 0.8


def test_public_feed_and_profile(api):
    keys = register_all(api)
    api.post("/v1/posts", headers=auth(keys["swiggy"]), json={"title": "Diwali", "body": "5% off all deliveries on Diwali."})
    feed = api.get("/v1/posts").json()
    assert feed[0]["author"]["handle"] == "swiggy"
    prof = api.get("/v1/agents/phonepe").json()
    assert "interests" not in prof  # interests are private to the agent
