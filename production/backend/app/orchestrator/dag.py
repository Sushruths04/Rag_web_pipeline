"""Stage graph: definition + validation + ordering. No execution here."""
from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Callable, Optional


@dataclass(frozen=True)
class StageDef:
    name: str
    label: str
    deps: tuple[str, ...] = ()
    track: str = "gt"
    fn: Optional[Callable] = None  # bound by the stage registry at run time


class PipelineDAG:
    def __init__(self, stages: list[StageDef]) -> None:
        names = [s.name for s in stages]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate stage names: {sorted(dupes)}")
        self._by_name = {s.name: s for s in stages}
        for s in stages:
            unknown = set(s.deps) - self._by_name.keys()
            if unknown:
                raise ValueError(f"stage {s.name!r} has unknown deps: {sorted(unknown)}")
        try:
            sorter = TopologicalSorter({s.name: set(s.deps) for s in stages})
            self._topo = list(sorter.static_order())
        except CycleError as e:
            raise ValueError(f"cycle in stage graph: {e.args[1]}") from e

    @property
    def names(self) -> list[str]:
        return list(self._by_name)

    def stage(self, name: str) -> StageDef:
        return self._by_name[name]

    def topo_order(self) -> list[str]:
        return list(self._topo)

    def downstream(self, name: str) -> set[str]:
        out: set[str] = set()
        frontier = {name}
        while frontier:
            nxt = {
                s.name
                for s in self._by_name.values()
                if frontier & set(s.deps) and s.name not in out
            }
            out |= nxt
            frontier = nxt
        return out
