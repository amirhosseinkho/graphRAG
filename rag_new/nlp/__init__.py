# -*- coding: utf-8 -*-
"""
NLP - پردازش زبان طبیعی
"""
from .search import Dealer, index_name
from .query import FulltextQueryer

__all__ = [
    "Dealer",
    "index_name",
    "FulltextQueryer"
] 