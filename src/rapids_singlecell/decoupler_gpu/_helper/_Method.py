from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from rapids_singlecell.decoupler_gpu._helper._run import _run

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapids_singlecell.decoupler_gpu._helper._data import DataType


class MethodMeta:
    def __init__(
        self,
        name: str,
        desc: str,
        func: Callable,
        *,
        stype: str,
        adj: bool,
        weight: bool,
        test: bool,
        limits: tuple,
        reference: str,
    ):
        self.name = name
        self.desc = desc
        self.func = func
        self.stype = stype
        self.adj = adj
        self.weight = weight
        self.test = test
        self.limits = limits
        self.reference = reference

    def meta(self) -> pd.DataFrame:
        meta = pd.DataFrame(
            [
                {
                    "name": self.name,
                    "desc": self.desc,
                    "stype": self.stype,
                    "weight": self.weight,
                    "test": self.test,
                    "limits": self.limits,
                    "reference": self.reference,
                }
            ]
        )
        return meta


class Method(MethodMeta):
    def __init__(
        self,
        _method: MethodMeta,
    ):
        super().__init__(
            name=_method.name,
            desc=_method.desc,
            func=_method.func,
            stype=_method.stype,
            adj=_method.adj,
            weight=_method.weight,
            test=_method.test,
            limits=_method.limits,
            reference=_method.reference,
        )
        self._method = _method
        self.__doc__ = self.func.__doc__

    def __call__(
        self,
        data: DataType,
        net: pd.DataFrame,
        *,
        tmin: int | float = 5,
        raw: bool = False,
        empty: bool = True,
        bsize: int | float = 5000,
        verbose: bool = False,
        pre_load: bool = False,
        **kwargs,
    ):
        return _run(
            name=self.name,
            func=self.func,
            adj=self.adj,
            test=self.test,
            data=data,
            net=net,
            tmin=tmin,
            raw=raw,
            empty=empty,
            bsize=bsize,
            verbose=verbose,
            pre_load=pre_load,
            **kwargs,
        )
