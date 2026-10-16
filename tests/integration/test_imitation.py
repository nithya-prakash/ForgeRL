from factory_scheduler.config import load_scenarios
from factory_scheduler.environment import features as F
from factory_scheduler.policies.heuristics import get_policy
from factory_scheduler.rl.imitation import clone, collect, new_model


def test_cloning_reduces_loss_and_only_keeps_visible_actions():
    scen = {"stable": load_scenarios()["stable"].replace(n_jobs=8)}
    data = collect(get_policy("ATC"), scen, [0, 1, 2])
    obs, masks, acts = data
    assert len(acts) > 50 and acts.max() < F.N_SLOTS
    assert masks[range(len(acts)), acts].all()  # teacher action is always a legal slot
    model = new_model(scen, [0], {}, seed=0)
    losses = clone(model, data, epochs=15)
    assert losses[-1] < 0.7 * losses[0]
