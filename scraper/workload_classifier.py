"""
Workload classifier: maps K8s pod/container/namespace metadata to workload categories.

Rules are loaded from config.yaml but can be extended programmatically.
The classifier is intentionally simple — fnmatch patterns on pod/container names.
"""

import fnmatch
import re
from dataclasses import dataclass


@dataclass
class WorkloadRule:
    category: str
    container_patterns: list[str]
    namespace_patterns: list[str]


class WorkloadClassifier:
    def __init__(self, rules: list[dict]):
        self.rules = []
        for rule in rules:
            match = rule.get("match", {})
            self.rules.append(WorkloadRule(
                category=rule["category"],
                container_patterns=[p.lower() for p in match.get("container_patterns", [])],
                namespace_patterns=[p.lower() for p in match.get("namespace_patterns", [])],
            ))

    def classify(self, namespace: str, pod: str, container: str) -> str:
        """Classify a workload based on K8s metadata. Returns category string."""
        if not pod and not container:
            return "idle"

        ns = (namespace or "").lower()
        pod_lower = (pod or "").lower()
        ctr_lower = (container or "").lower()

        for rule in self.rules:
            # Check namespace patterns
            ns_match = not rule.namespace_patterns or any(
                fnmatch.fnmatch(ns, p) for p in rule.namespace_patterns
            )
            # Check container/pod patterns (match against both)
            ctr_match = not rule.container_patterns or any(
                fnmatch.fnmatch(ctr_lower, p) or fnmatch.fnmatch(pod_lower, p)
                for p in rule.container_patterns
            )

            if ns_match and ctr_match and rule.category != "other":
                return rule.category

            # For "other" category with wildcard, only match if there's actually a container
            if rule.category == "other" and ctr_match and container:
                return rule.category

        return "idle"
