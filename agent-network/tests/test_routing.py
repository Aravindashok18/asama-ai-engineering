from agentnet.models import AgentUpdate, PostCreate
from agentnet.relevance import feedback_lift, intent_match, topic_overlap
from agentnet.taxonomy import TopicClassifier


def route(platform, agents, author, title, body):
    post = platform.create_post(agents[author]["agent"], PostCreate(title=title, body=body))
    ranked = platform.route_post(post["id"])
    handles = {v["agent"]["id"]: h for h, v in agents.items()}
    return post, {handles[c.agent_id]: c for c in ranked}


def notified(cands):
    return {h for h, c in cands.items() if c.decision == "notify"}


def test_classifier_reads_offer_and_festival():
    topics = TopicClassifier().classify("5% off all deliveries on Diwali.")
    assert {"discount_offers", "festive_season"} <= set(topics)


def test_diwali_example_from_the_brief(seeded):
    platform, agents = seeded
    post, cands = route(platform, agents, "swiggy", "Diwali delivery offer", "5% off all deliveries on Diwali.")
    assert "food_delivery" in post["topics"]  # inherited from the author's industry
    assert {"phonepe", "hdfc_bank"} <= notified(cands)
    assert "bajaj_auto" not in notified(cands)
    assert "swiggy" not in cands  # never route to the author


def test_vehicle_launch_goes_to_lenders_and_insurers_not_food(seeded):
    platform, agents = seeded
    _, cands = route(platform, agents, "bajaj_auto", "Pulsar festive price cut",
                     "Bajaj cuts Pulsar prices for the festive season and wants auto loan and motor insurance partners.")
    got = notified(cands)
    assert {"hdfc_bank", "tata_aig"} <= got
    assert not got & {"swiggy", "zomato", "delhivery"}


def test_every_candidate_is_logged_with_features(seeded):
    platform, agents = seeded
    post, cands = route(platform, agents, "swiggy", "Diwali delivery offer", "5% off all deliveries on Diwali.")
    log = platform.store.routing_log(post["id"])
    assert len(log) == len(cands)
    assert {"semantic", "topic_overlap", "intent_match", "exclusion", "feedback_lift"} <= set(log[0]["features"])


def test_follow_forces_delivery_and_mute_blocks_it(seeded):
    platform, agents = seeded
    bajaj = agents["bajaj_auto"]["agent"]
    platform.store.add_subscription(bajaj["id"], "agent", agents["swiggy"]["agent"]["id"], "follow")
    _, cands = route(platform, agents, "swiggy", "Diwali delivery offer", "5% off all deliveries on Diwali.")
    assert cands["bajaj_auto"].decision == "notify"
    assert cands["bajaj_auto"].reason == "explicit follow"

    phonepe = agents["phonepe"]["agent"]
    platform.store.add_subscription(phonepe["id"], "topic", "food_delivery", "mute")
    _, cands = route(platform, agents, "swiggy", "Holi offer", "10% off all deliveries on Holi.")
    assert "phonepe" not in cands


def test_keyword_subscription(seeded):
    platform, agents = seeded
    croma = agents["croma"]["agent"]
    platform.store.add_subscription(croma["id"], "keyword", "big billion days", "follow")
    _, cands = route(platform, agents, "flipkart", "Sale dates", "Big Billion Days starts on the 1st.")
    assert cands["croma"].decision == "notify"
    assert "follow:keyword" in cands["croma"].sources


def test_fanout_cap_and_hourly_budget(seeded):
    platform, agents = seeded
    platform.settings.max_fanout = 1
    _, cands = route(platform, agents, "swiggy", "Diwali delivery offer", "5% off all deliveries on Diwali.")
    assert len(notified(cands)) == 1
    platform.settings.max_fanout = 50

    phonepe = agents["phonepe"]["agent"]
    platform.update_agent(phonepe, AgentUpdate(max_notifications_per_hour=1))
    _, cands = route(platform, agents, "swiggy", "Holi offer", "10% off all deliveries on Holi.")
    assert cands["phonepe"].decision == "skip"
    assert "budget" in cands["phonepe"].reason


def test_unverified_authors_reach_fewer_agents(platform):
    from agentnet.seed import seed_platform

    agents = seed_platform(platform, verify=False)
    _, cands = route(platform, agents, "swiggy", "Diwali delivery offer", "5% off all deliveries on Diwali.")
    unverified = {h: c.score for h, c in cands.items()}
    assert all(c.decision == "skip" or c.score >= platform.settings.unverified_threshold for c in cands.values())
    assert unverified  # still scored, just held to a higher bar


def test_exploration_is_deterministic_and_bounded(seeded):
    platform, agents = seeded
    platform.settings.exploration_rate = 1.0
    _, cands = route(platform, agents, "swiggy", "Diwali delivery offer", "5% off all deliveries on Diwali.")
    explored = [c for c in cands.values() if c.exploration]
    thr = platform.settings.threshold
    assert all(thr - platform.settings.exploration_band <= c.score < thr for c in explored)


def test_feature_functions():
    assert topic_overlap({"a": 1.0}, {}) == 0.0
    assert topic_overlap({"a": 1.0, "b": 1.0}, {"a": 1.0}) == 1.0  # extra post topics do not dilute
    assert intent_match({"a": 1, "b": 1}, [{"a", "b", "c"}]) == 1.0
    assert intent_match({"a": 1}, [{"a", "b"}]) == 0.5
    assert feedback_lift({}, ["topic:x"]) == 0.0
    assert feedback_lift({"topic:x": (1.0, 11.0)}, ["topic:x"]) < -0.2
    assert feedback_lift({"topic:x": (11.0, 1.0)}, ["topic:x"]) > 0.2


def test_llm_judge_overrides_only_the_borderline_band(seeded):
    platform, agents = seeded
    calls = []

    def judge(post_text, profile):
        calls.append(profile)
        return False

    platform.router.judge = judge
    platform.settings.llm_judge_band = 0.6  # everything is "borderline" for this test
    _, cands = route(platform, agents, "swiggy", "Diwali delivery offer", "5% off all deliveries on Diwali.")
    assert calls
    assert not notified(cands)
    assert all("llm judge" in c.reason for c in cands.values())


def test_judge_failure_falls_back_to_score(seeded):
    platform, agents = seeded

    def broken(post_text, profile):
        raise TimeoutError("model down")

    platform.router.judge = broken
    platform.settings.llm_judge_band = 0.6
    _, cands = route(platform, agents, "swiggy", "Diwali delivery offer", "5% off all deliveries on Diwali.")
    assert {"phonepe", "hdfc_bank"} <= notified(cands)
