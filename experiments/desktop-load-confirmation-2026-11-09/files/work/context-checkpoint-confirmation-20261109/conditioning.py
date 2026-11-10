"""Bounded stable-nominal preparation, with no GPU/inference/sample retries."""
from quiet_preparation import wait_nominal,QuietPreparationFailure


class ConditioningFailure(RuntimeError):
    pass


def settle_bounded(events, *, label, stable_seconds, max_seconds, observe,
                   thermal, check, persist, **testing_clock):
    if max_seconds!=1200 or stable_seconds!=15:
        raise ValueError('Qualification preparation is fixed at15s nominal with1200s wall cap')
    try:
        return wait_nominal(events,context={'label':label,'purpose':'qualification_preparation'},
            stable_seconds=stable_seconds,max_seconds=max_seconds,observe=observe,
            thermal=thermal,check=check,persist=persist,**testing_clock)
    except QuietPreparationFailure as error:
        raise ConditioningFailure(str(error)) from error
