# -*- coding: utf-8 -*-
"""
Entity Resolution - حل موجودیت‌های مشابه (با ادغام امن و جلوگیری از over-merge)
"""
import logging
import unicodedata
import re
from copy import deepcopy
from typing import Dict, List, Any, Optional, Tuple, Iterable
import networkx as nx

class EntityResolution:
    """کلاس برای حل موجودیت‌های مشابه در گراف"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.resolved_entities = {}
        self._blocked_pairs: List[Tuple[str, str]] = []
        # واژه‌های مرتبط ولی متمایز که نباید merge شوند
        self._related_but_distinct = {
            ("cancer", "tumor"), ("neoplasm", "tumor"),
            ("mapk", "mapk1"), ("mapk", "mapk3"),
            ("her2", "erbb2"),  # توجه: بسته به رجیستری ممکن است یکسان باشند؛ در صورت داشتن نگاشت رسمی، از لیست حذف کنید
        }

    # -------------------- Normalization & Similarity --------------------
    def _norm(self, s: str) -> str:
        if not s:
            return ""
        x = unicodedata.normalize("NFKC", str(s)).lower()
        # Greek letters → latin tokens
        greek_map = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "κ": "kappa", "μ": "mu", "π": "pi"}
        for k, v in greek_map.items():
            x = x.replace(k, v)
        # Roman numerals (basic) → arabic
        x = f" {x} "
        roman_map = {" ii ": " 2 ", " iii ": " 3 ", " iv ": " 4 ", " vi ": " 6 ", " vii ": " 7 ", " ix ": " 9 ", " x ": " 10 "}
        for k, v in roman_map.items():
            x = x.replace(k, v)
        # Remove punctuation-like separators
        x = re.sub(r"[\(\)\[\]\{\},.;:/\\_\-]+", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x

    def _jaccard_chars(self, a: str, b: str) -> float:
        A, B = set(a), set(b)
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def _lcs_len(self, a: str, b: str) -> int:
        # O(n*m) dynamic programming for short strings
        n, m = len(a), len(b)
        dp = [0] * (m + 1)
        for i in range(1, n + 1):
            prev = 0
            for j in range(1, m + 1):
                cur = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = dp[j] if dp[j] > dp[j - 1] else dp[j - 1]
                prev = cur
        return dp[m]
    
    def calculate_similarity(self, entity1: str, entity2: str) -> float:
        """محاسبه شباهت بین دو موجودیت (نرمال‌سازی + Jaccard/LCS)"""
        try:
            e1 = self._norm(entity1)
            e2 = self._norm(entity2)
            if not e1 or not e2:
                return 0.0
            if e1 == e2:
                return 1.0
            j = self._jaccard_chars(e1, e2)
            lcs = self._lcs_len(e1, e2) / max(len(e1), len(e2))
            sim = 0.55 * j + 0.45 * lcs
            return float(max(0.0, min(1.0, sim)))
        except Exception as e:
            logging.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    # -------------------- Bucketing & Constraints --------------------
    def _node_type_ns(self, G: nx.Graph, n: Any) -> Tuple[Optional[str], Optional[str]]:
        attrs = G.nodes[n]
        t = attrs.get("type") or attrs.get("kind")
        ns = attrs.get("namespace")
        if ns is None and isinstance(n, str) and "::" in n:
            ns = n.split("::", 1)[0]
        return (t, ns)

    def _same_bucket(self, G: nx.Graph, n1: Any, n2: Any) -> bool:
        t1, ns1 = self._node_type_ns(G, n1)
        t2, ns2 = self._node_type_ns(G, n2)
        if t1 and t2 and t1 != t2:
            return False
        if ns1 and ns2 and ns1 != ns2:
            return False
        return True

    def _block_pair(self, a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        return (a, b) in self._related_but_distinct or (b, a) in self._related_but_distinct

    def _label(self, G: nx.Graph, n: Any) -> str:
        return str(G.nodes[n].get("name") or n)

    # -------------------- Edge Redirection (merge-safe) --------------------
    def _redirect_edges(self, G: nx.Graph, src: Any, dst: Any) -> None:
        """انتقال یال‌ها از src به dst با حفظ ویژگی‌ها و جهت/کلیدها."""
        if src == dst or src not in G or dst not in G:
            return
        try:
            if isinstance(G, nx.MultiDiGraph):
                for u, _, k, data in list(G.in_edges(src, keys=True, data=True)):
                    if u == dst:
                        continue
                    if not G.has_edge(u, dst, key=k):
                        G.add_edge(u, dst, key=k, **(data or {}))
                for _, v, k, data in list(G.out_edges(src, keys=True, data=True)):
                    if v == dst:
                        continue
                    if not G.has_edge(dst, v, key=k):
                        G.add_edge(dst, v, key=k, **(data or {}))
            elif isinstance(G, nx.DiGraph):
                for u in list(G.predecessors(src)):
                    if u == dst:
                        continue
                    data = deepcopy(G.get_edge_data(u, src))
                    if not G.has_edge(u, dst):
                        G.add_edge(u, dst, **(data or {}))
                for v in list(G.successors(src)):
                    if v == dst:
                        continue
                    data = deepcopy(G.get_edge_data(src, v))
                    if not G.has_edge(dst, v):
                        G.add_edge(dst, v, **(data or {}))
            elif isinstance(G, nx.MultiGraph):
                for u in list(G.neighbors(src)):
                    if u == dst:
                        continue
                    datas = G.get_edge_data(u, src) or {}
                    for k, data in list(datas.items()):
                        if not G.has_edge(u, dst, key=k):
                            G.add_edge(u, dst, key=k, **(data or {}))
            else:  # undirected simple Graph
                for u in list(G.neighbors(src)):
                    if u == dst:
                        continue
                    data = deepcopy(G.get_edge_data(u, src))
                    if not G.has_edge(u, dst):
                        G.add_edge(u, dst, **(data or {}))
        except Exception as e:
            logging.warning(f"Edge redirection failed for {src}->{dst}: {e}")
    
    def find_similar_entities(self, entities: List[str]) -> List[List[str]]:
        """یافتن گروه‌های موجودیت‌های مشابه"""
        groups = []
        processed = set()
        
        for i, entity1 in enumerate(entities):
            if entity1 in processed:
                continue
            
            group = [entity1]
            processed.add(entity1)
            
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity2 in processed:
                    continue
                
                similarity = self.calculate_similarity(entity1, entity2)
                if similarity >= self.similarity_threshold:
                    group.append(entity2)
                    processed.add(entity2)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def resolve_entities_in_graph(self, G: nx.Graph, dry_run: bool = False) -> nx.Graph:
        """
        حل موجودیت‌های مشابه در گراف با قیود نوع/namespace و ادغام امن.
        اگر dry_run=True باشد، صرفاً گروه‌ها شناسایی می‌شوند و گراف تغییر نمی‌کند.
        """
        try:
            nodes = list(G.nodes())
            processed = set()
            # بلوک‌بندی ساده بر اساس نوع/namespace برای کاهش هزینه
            buckets: Dict[Tuple[Optional[str], Optional[str]], List[Any]] = {}
            for n in nodes:
                key = self._node_type_ns(G, n)
                buckets.setdefault(key, []).append(n)

            for (_t, _ns), bucket_nodes in buckets.items():
                m = len(bucket_nodes)
                for i in range(m):
                    n1 = bucket_nodes[i]
                    if n1 in processed:
                        continue
                    group = [n1]
                    label1 = self._label(G, n1)
                    for j in range(i + 1, m):
                        n2 = bucket_nodes[j]
                        if n2 in processed:
                            continue
                        if not self._same_bucket(G, n1, n2):
                            continue
                        if self._block_pair(str(label1), str(self._label(G, n2))):
                            self._blocked_pairs.append((str(label1), str(self._label(G, n2))))
                            continue
                        sim = self.calculate_similarity(label1, self._label(G, n2))
                        if sim >= self.similarity_threshold:
                            group.append(n2)
                            processed.add(n2)

                    if len(group) <= 1:
                        continue

                    # انتخاب نماینده: اولویت با داشتن id/namespace، سپس درجه بیشتر
                    def rep_key(n):
                        nid = G.nodes[n].get("id") or None
                        ns = G.nodes[n].get("namespace") or None
                        return (1 if (nid or ns) else 0, G.degree(n))

                    representative = max(group, key=rep_key)

                    if dry_run:
                        self.resolved_entities[representative] = group
                        continue

                    # ادغام ویژگی‌ها با set/merge امن
                    merged_attrs: Dict[str, Any] = {}
                    for node in group:
                        if node not in G.nodes:
                            continue
                        node_attrs = G.nodes[node]
                        for key, value in node_attrs.items():
                            if key not in merged_attrs:
                                merged_attrs[key] = deepcopy(value)
                            else:
                                # در تعارض، مقدار نماینده غالب است
                                if node == representative:
                                    merged_attrs[key] = deepcopy(value)
                                else:
                                    if isinstance(value, list):
                                        base = merged_attrs.get(key, [])
                                        if not isinstance(base, list):
                                            base = [base]
                                        merged_attrs[key] = list(set(base) | set(value))
                                    elif isinstance(value, dict) and isinstance(merged_attrs.get(key), dict):
                                        merged_attrs[key].update(value)
                                    # دیگر انواع: مقدار موجود حفظ می‌شود

                    # به‌روزرسانی نود نماینده
                    G.nodes[representative].update(merged_attrs)

                    # انتقال یال‌ها و حذف نودهای ادغام شده
                    for node in group:
                        if node == representative:
                            continue
                        if node not in G:
                            continue
                        self._redirect_edges(G, node, representative)
                        if node in G:
                            G.remove_node(node)

                    self.resolved_entities[representative] = group

            return G

        except Exception as e:
            logging.error(f"Error in resolve_entities_in_graph: {e}")
            return G
    
    def get_resolution_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه عملیات حل موجودیت"""
        return {
            "resolved_groups": len(self.resolved_entities),
            "total_resolved_entities": sum(len(group) for group in self.resolved_entities.values()),
            "representatives": list(self.resolved_entities.keys()),
            "resolution_mapping": self.resolved_entities,
            "blocked_pairs": self._blocked_pairs
        }
    
    def clear_resolution_cache(self):
        """پاک کردن کش حل موجودیت"""
        self.resolved_entities.clear() 