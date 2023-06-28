from tests.e2e import assert_rank_corr


def test_rank_correlation(joint_sample, rank_corr, rank_corr_measure):
    assert_rank_corr(joint_sample, rank_corr, rank_corr_measure, rel=1 / 100, abs=0)
