# ================================================================
# Graph Forecaster conceptual code as an add-on to LLM
# February 14 2026 Dmitry Feklin FeklinDN@gmail.com
# pip install biopython crossrefapi scholarly requests sentence-transformers duckduckgo_search plotly

import os
import json
import uuid
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
import re
import logging

# ML & Embeddings
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import networkx as nx

# LLM
from openai import OpenAI

# Data sources (core)
from duckduckgo_search import DDGS
import arxiv
import requests
from bs4 import BeautifulSoup

# Visualization
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Progress tracking
from tqdm import tqdm

# ----------------------------------------------------------------
# ОБЯЗАТЕЛЬНЫЕ БИБЛИОТЕКИ — без них запуск невозможен
# ----------------------------------------------------------------
# Logging нужен до exit(), поэтому настраиваем его здесь минимально
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('graph_forecaster.log'),
        logging.StreamHandler()
    ]
)
_pre_logger = logging.getLogger("startup")

try:
    from Bio import Entrez
    from scholarly import scholarly
    from crossref.restful import Works
    BIOPYTHON_AVAILABLE  = True
    SCHOLARLY_AVAILABLE  = True
    CROSSREF_AVAILABLE   = True
except ImportError as e:
    _pre_logger.error(
        f"Missing required library: {e}. "
        "Install with: pip install biopython scholarly crossrefapi"
    )
    exit(1)

# ----------------------------------------------------------------
# ОПЦИОНАЛЬНЫЕ БИБЛИОТЕКИ — функции-заглушки активируются при отсутствии
# ----------------------------------------------------------------
try:
    import transformers                    # для sentiment-моделей
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    _pre_logger.warning("transformers не установлен — sentiment-анализ отключён. pip install transformers torch")

# ================================================================
# LOGGING (уже настроен в блоке импортов выше)
# ================================================================
logger = logging.getLogger(__name__)

# ================================================================
# CONFIG
# ================================================================
class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    CRUNCHBASE_API_KEY   = os.getenv("CRUNCHBASE_API_KEY",   "")   # заглушка
    PITCHBOOK_API_KEY    = os.getenv("PITCHBOOK_API_KEY",    "")   # заглушка
    DEALROOM_API_KEY     = os.getenv("DEALROOM_API_KEY",     "")   # заглушка
    REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "")   # заглушка
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")   # заглушка
    
    # Models
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-4o-mini"

    # Sentiment models (опциональные, загружаются лениво)
    # Модель на рецензиях (фильмы/книги) — для восприятия технологии «широкой аудиторией»
    SENTIMENT_REVIEW_MODEL  = "nlptown/bert-base-multilingual-uncased-sentiment"
    # Модель на художественных текстах — для «нарративного» восприятия темы
    SENTIMENT_FICTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
    
    # Search limits per source
    MAX_ARXIV_PER_QUERY          = 20
    MAX_SCHOLAR_PER_QUERY        = 15
    MAX_CROSSREF_PER_QUERY       = 15
    MAX_PUBMED_PER_QUERY         = 15
    MAX_SEMANTIC_SCHOLAR_PER_QUERY = 20
    MAX_WEB_PER_QUERY            = 15
    MAX_PATENTS_PER_QUERY        = 10
    MAX_GITHUB_PER_QUERY         = 10
    MAX_RESEARCHGATE_PER_QUERY   = 10   # ResearchGate (web scraping)
    MAX_FORUM_PER_QUERY          = 20   # форумы (Reddit, HN, StackExchange)
    MAX_INVESTMENT_PER_QUERY     = 10   # финансовые базы (заглушки)
    
    # Discovery parameters
    MAX_RECURSIVE_DEPTH   = 5
    MAX_NODES_PER_LEVEL   = 50
    CROSS_DOMAIN_BATCH_SIZE = 20
    
    # Similarity thresholds
    MIN_SEMANTIC_SIMILARITY = 0.15
    MIN_EDGE_CONFIDENCE     = 0.10
    
    # Rate limiting
    REQUESTS_PER_MINUTE     = 60
    DELAY_BETWEEN_REQUESTS  = 1.0
    
    # Cache
    CACHE_DIR    = "./cache"
    ENABLE_CACHE = True
    
    # Weights
    EDGE_WEIGHTS = {
        'semantic':             0.30,
        'temporal':             0.15,
        'limitation_resolution': 0.25,
        'citation':             0.10,
        'investment':           0.10,
        'social':               0.10
    }
    
    READINESS_WEIGHTS = {
        'scientific':  0.25,
        'investment':  0.25,
        'social':      0.20,
        'maturity':    0.15,
        'group_size':  0.15
    }

# ================================================================
# DATA STRUCTURES (Same as before)
# ================================================================

@dataclass
class Node:
    """Enhanced node with comprehensive metadata"""
    id: str
    node_type: str
    timestamp: float
    embedding: np.ndarray
    
    # Core
    description: str
    full_text: str = ""
    title: str = ""
    
    # Analysis
    advantages: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    mentioned_domains: List[str] = field(default_factory=list)  # NEW: domains mentioned in text
    cited_works: List[str] = field(default_factory=list)  # NEW: referenced papers/technologies
    
    # Metrics (raw)
    scientific_citations: int = 0
    patent_citations: int = 0
    investment_usd: float = 0.0
    social_mentions: int = 0
    media_coverage: int = 0
    github_stars: int = 0
    
    # Metrics (normalized 0-10)
    scientific_score: float = 0.0
    investment_score: float = 0.0
    social_score: float = 0.0
    maturity_score: float = 0.0
    
    # Derived
    group_id: int = -1
    group_count: int = 1
    readiness_score: float = 0.0
    convergence_potential: float = 0.0
    
    # Relations
    solves_limitations: List[str] = field(default_factory=list)
    requires_nodes: List[str] = field(default_factory=list)
    enables_nodes: List[str] = field(default_factory=list)
    
    # Metadata
    source_urls: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    geographic_focus: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    
    # Discovery tracking
    discovery_depth: int = 0
    discovery_path: List[str] = field(default_factory=list)
    discovery_query: str = ""
    
    # Extended metrics for acceleration / drag analysis
    dual_use_risk: float = 0.0
    strategic_value: float = 0.0
    legal_risk_score: float = 0.0
    export_control_risk: float = 0.0

    # Social perception (заполняется sentiment-моделями)
    sentiment_review_score: float = 0.0    # восприятие через рецензии (0-1, где 1 = позитивно)
    sentiment_fiction_score: float = 0.0   # нарративное восприятие (emotion score)
    sentiment_forum_score: float = 0.0     # тональность обсуждений на форумах
    social_perception_score: float = 0.0  # агрегированный индекс социального восприятия

    # Investment / financial data (заполняется заглушками или реальными коннекторами)
    investment_rounds: int = 0             # количество раундов финансирования
    investment_total_usd: float = 0.0      # суммарный объём инвестиций, USD
    investment_last_round_usd: float = 0.0 # последний раунд, USD
    investment_lead_investors: List[str] = field(default_factory=list)
    investment_data_source: str = ""       # откуда взяты данные ("crunchbase", "pitchbook", ...)

    # Forum signals (заполняется _search_forums)
    forum_post_count: int = 0
    forum_sentiment_raw: List[float] = field(default_factory=list)  # сырые оценки постов

    # Temporal zone attributes
    is_temporal_zone: bool = False
    zone_multiplier: float = 1.0
    contained_nodes: List[str] = field(default_factory=list)
    acceleration_multiplier: float = 1.0

    # Structural (populated by compute_structural_dependencies)
    structural_dependency_index: float = 0.0
    cascade_influence: float = 0.0
    upstream_pressure: float = 0.0

    # Forecast
    forecast_score: float = 0.0

    # Processing flags
    processed: bool = False
    expanded: bool = False
    domains_extracted: bool = False

@dataclass
class Edge:
    """Multi-dimensional weighted edge"""
    source: str
    target: str
    
    # Weight components
    semantic_similarity: float = 0.0
    temporal_proximity: float = 0.0
    limitation_resolution: float = 0.0
    citation_link: float = 0.0
    investment_correlation: float = 0.0
    social_correlation: float = 0.0
    
    # Derived
    total_weight: float = 0.0
    confidence: float = 0.0
    
    # Evidence
    evidence: List[str] = field(default_factory=list)
    relationship_type: str = "related"
    
    # Discovery
    discovery_method: str = "direct"

# ================================================================
# CACHE & RATE LIMITER (Same as before)
# ================================================================

class Cache:
    """Persistent cache for API calls"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.embeddings_cache = os.path.join(cache_dir, "embeddings.pkl")
        self.llm_cache = os.path.join(cache_dir, "llm_responses.pkl")
        self.search_cache = os.path.join(cache_dir, "search_results.pkl")
        
        self.embeddings = self._load(self.embeddings_cache)
        self.llm_responses = self._load(self.llm_cache)
        self.search_results = self._load(self.search_cache)
    
    def _load(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save(self, data: dict, path: str):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        key = hashlib.md5(text.encode()).hexdigest()
        return self.embeddings.get(key)
    
    def set_embedding(self, text: str, embedding: np.ndarray):
        key = hashlib.md5(text.encode()).hexdigest()
        self.embeddings[key] = embedding
        self._save(self.embeddings, self.embeddings_cache)
    
    def get_llm_response(self, prompt: str, model: str) -> Optional[str]:
        key = hashlib.md5((prompt + model).encode()).hexdigest()
        return self.llm_responses.get(key)
    
    def set_llm_response(self, prompt: str, model: str, response: str):
        key = hashlib.md5((prompt + model).encode()).hexdigest()
        self.llm_responses[key] = response
        self._save(self.llm_responses, self.llm_cache)
    
    def get_search_results(self, query: str, source: str) -> Optional[list]:
        key = hashlib.md5((query + source).encode()).hexdigest()
        return self.search_results.get(key)
    
    def set_search_results(self, query: str, source: str, results: list):
        key = hashlib.md5((query + source).encode()).hexdigest()
        self.search_results[key] = results
        self._save(self.search_results, self.search_cache)

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.rate = requests_per_minute / 60.0
        self.tokens = requests_per_minute
        self.max_tokens = requests_per_minute
        self.last_update = time.time()
    
    def wait_if_needed(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens < 1:
            wait_time = (1 - self.tokens) / self.rate
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
            self.tokens = 1
        
        self.tokens -= 1

# ================================================================
# GRAPH FORECASTER V6 - CORRECTED
# ================================================================

class GraphForecasterV6:
    """
    Exhaustive graph builder with intelligent cross-domain discovery
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        
        # Initialize LLM and embedder
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Data structures
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        
        # Tracking
        self.query_history: Set[str] = set()
        self.processed_urls: Set[str] = set()
        self.discovered_domains: Set[str] = set()  # NEW: track discovered domains
        
        # Utilities
        self.cache = Cache(config.CACHE_DIR) if config.ENABLE_CACHE else None
        self.rate_limiter = RateLimiter(config.REQUESTS_PER_MINUTE)
        self.scaler = MinMaxScaler()
        
        # Crossref client
        self.crossref_works = Works()
        
        # State
        self.target_problem = ""
        self.problem_tree = {}

        # Sentiment pipelines (загружаются лениво при первом вызове)
        self._sentiment_review_pipe  = None   # nlptown review model
        self._sentiment_fiction_pipe = None   # emotion / fiction model

        logger.info("Graph Forecaster V6 (Corrected) initialized")
    
    # ================================================================
    # STAGE 0: Problem Decomposition (Same as before)
    # ================================================================
    
    def define_target(self, target: str) -> Dict:
        """Deep problem decomposition with multi-level analysis"""
        logger.info(f"STAGE 0: Defining target problem")
        logger.info(f"Target: {target}")
        
        self.target_problem = target
        
        prompt = f"""
        Perform deep analysis of this technology target:
        
        TARGET: {target}
        
        Provide comprehensive JSON:
        {{
          "core_concept": "One-sentence description",
          "key_advantages": ["advantage1", "advantage2", ...],
          "known_limitations": [
            {{
              "limitation": "specific technical problem",
              "severity": "critical|high|medium|low",
              "category": "cost|speed|scalability|reliability|adoption|regulatory",
              "why_matters": "impact explanation"
            }},
            ...
          ],
          "current_maturity": "concept|research|prototype|early_product|mature",
          "primary_applications": ["app1", "app2", ...],
          "competing_approaches": ["approach1", "approach2", ...],
          "required_breakthroughs": ["breakthrough1", ...],
          "adjacent_fields": ["field1", "field2", ...],
          "key_players": ["organization1", ...]
        }}
        
        Be thorough - this will guide all subsequent searches.
        """
        
        response = self._call_llm(prompt, temperature=0.2, response_format="json")
        self.problem_tree = json.loads(response)
        
        logger.info(f"Problem tree created with {len(self.problem_tree.get('known_limitations', []))} limitations")
        
        return self.problem_tree
    
    # ================================================================
    # STAGE 1: MULTI-SOURCE INGESTION (EXPANDED)
    # ================================================================
    
    def _auto_connect_nodes(self, threshold=0.48):
        nodes = list(self.nodes.values())
        embeddings = np.array([n.embedding for n in nodes])
        sim_matrix = cosine_similarity(embeddings)
    
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if sim_matrix[i,j] > threshold:
                    edge = Edge(source=nodes[i].id, target=nodes[j].id, 
                              semantic_similarity=sim_matrix[i,j],
                              confidence=sim_matrix[i,j])
                    self.edges.append(edge)

    def ingest_all_sources(self, depth: int = 3):
        """
        Comprehensive ingestion from ALL available sources
        """
        logger.info(f"STAGE 1: Multi-source exhaustive ingestion (depth={depth})")
        
        # Level 0: Direct queries about target
        level_0_queries = self._generate_comprehensive_queries(self.target_problem, "direct")
        logger.info(f"Level 0: {len(level_0_queries)} queries generated")
        
        level_0_nodes = self._execute_multi_source_queries(level_0_queries, depth=0)
        logger.info(f"Level 0: {len(level_0_nodes)} nodes collected")
        
        # Extract domains from Level 0 nodes
        self._extract_domains_from_nodes(level_0_nodes)
        
        # Recursive expansion
        all_new_nodes = level_0_nodes
        
        for current_depth in range(1, depth + 1):
            logger.info(f"\n--- Expanding to depth {current_depth} ---")
            
            previous_nodes = [n for n in self.nodes.values() if n.discovery_depth == current_depth - 1]
            
            if not previous_nodes:
                logger.info(f"No nodes at depth {current_depth - 1}, stopping")
                break
            
            # Limit expansion to most promising nodes
            previous_nodes = sorted(previous_nodes, key=lambda n: n.convergence_potential, reverse=True)
            previous_nodes = previous_nodes[:self.config.MAX_NODES_PER_LEVEL]
            
            new_nodes_this_level = []
            
            for node in tqdm(previous_nodes, desc=f"Depth {current_depth}"):
                if node.expanded:
                    continue
                
                # Generate expansion queries from this node
                expansion_queries = self._generate_expansion_queries(node)
                
                # Execute across all sources
                new_nodes = self._execute_multi_source_queries(
                    expansion_queries,
                    depth=current_depth,
                    parent_node=node
                )
                
                new_nodes_this_level.extend(new_nodes)
                node.expanded = True
                
                # Extract domains from new nodes
                self._extract_domains_from_nodes(new_nodes)
                
                time.sleep(self.config.DELAY_BETWEEN_REQUESTS)
            
            logger.info(f"Depth {current_depth}: {len(new_nodes_this_level)} new nodes")
            all_new_nodes.extend(new_nodes_this_level)
            
            if len(new_nodes_this_level) < 5:
                logger.info(f"Too few new nodes at depth {current_depth}, stopping expansion")
                break
        
        logger.info(f"\nTotal nodes collected: {len(self.nodes)}")
        logger.info(f"Discovered domains: {len(self.discovered_domains)}")

        # Final domain extraction pass — catches any nodes added late or missed mid-loop
        unextracted = [n for n in self.nodes.values() if not n.domains_extracted]
        if unextracted:
            logger.info(f"Final domain extraction pass: {len(unextracted)} unprocessed nodes")
            self._extract_domains_from_nodes(unextracted)

        self._auto_connect_nodes()

        return all_new_nodes
    
    def _execute_multi_source_queries(self, queries: List[str], depth: int, parent_node: Optional[Node] = None) -> List[Node]:
        """
        Execute queries across ALL available sources
        """
        new_nodes = []
        
        for query in queries:
            if query in self.query_history:
                continue
            
            self.query_history.add(query)
            
            # Source 1: arXiv
            nodes_arxiv = self._search_arxiv(query, depth, parent_node)
            new_nodes.extend(nodes_arxiv)
            
            # Source 2: Semantic Scholar
            nodes_semantic = self._search_semantic_scholar(query, depth, parent_node)
            new_nodes.extend(nodes_semantic)
            
            # Source 3: Crossref (general academic)
            nodes_crossref = self._search_crossref(query, depth, parent_node)
            new_nodes.extend(nodes_crossref)
            
            # Source 4: PubMed (biomedical)
            nodes_pubmed = self._search_pubmed(query, depth, parent_node)
            new_nodes.extend(nodes_pubmed)
            
            # Source 5: Google Scholar (via scholarly)
            nodes_scholar = self._search_google_scholar(query, depth, parent_node)
            new_nodes.extend(nodes_scholar)
            
            # Source 6: Patents (Google Patents via web search)
            nodes_patents = self._search_patents(query, depth, parent_node)
            new_nodes.extend(nodes_patents)
            
            # Source 7: GitHub
            nodes_github = self._search_github(query, depth, parent_node)
            new_nodes.extend(nodes_github)
            
            # Source 8: General Web
            nodes_web = self._search_web(query, depth, parent_node)
            new_nodes.extend(nodes_web)

            # Source 9: ResearchGate (web scraping + link verification)
            nodes_rg = self._search_researchgate(query, depth, parent_node)
            new_nodes.extend(nodes_rg)

            # Source 10: Форумы (Reddit, HackerNews, StackExchange)
            nodes_forums = self._search_forums(query, depth, parent_node)
            new_nodes.extend(nodes_forums)

            # Source 11: Инвестиционные данные (заглушки)
            nodes_invest = self._fetch_investment_data(query, depth, parent_node)
            new_nodes.extend(nodes_invest)

            # Source 12: Память модели + верификация ссылок
            nodes_memory = self._search_model_memory(query, depth, parent_node)
            new_nodes.extend(nodes_memory)
            
            self.rate_limiter.wait_if_needed()
        
        return new_nodes
    
    # ================================================================
    # SEARCH IMPLEMENTATIONS - EXPANDED
    # ================================================================
    
    def _search_arxiv(self, query: str, depth: int, parent_node: Optional[Node]) -> List[Node]:
        """Search arXiv"""
        if self.cache:
            cached = self.cache.get_search_results(query, "arxiv")
            if cached:
                return self._nodes_from_cached_results(cached, "paper", depth, parent_node, query)
        
        nodes = []
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=self.config.MAX_ARXIV_PER_QUERY,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for result in search.results():
                url = result.entry_id
                
                if url in self.processed_urls:
                    continue
                
                self.processed_urls.add(url)
                
                text = f"Title: {result.title}\n\nAbstract: {result.summary}"
                
                node = self._create_node(
                    text=text,
                    url=url,
                    node_type="paper",
                    depth=depth,
                    parent_node=parent_node,
                    query=query,
                    metadata={
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'publication_date': result.published.isoformat(),
                        'categories': result.categories
                    }
                )
                
                nodes.append(node)
                results.append({
                    'title': result.title,
                    'abstract': result.summary,
                    'url': url,
                    'authors': [author.name for author in result.authors]
                })
            
            if self.cache:
                self.cache.set_search_results(query, "arxiv", results)
        
        except Exception as e:
            logger.error(f"arXiv search error for '{query}': {e}")
        
        return nodes
    
    def _search_semantic_scholar(self, query: str, depth: int, parent_node: Optional[Node]) -> List[Node]:
        """Search Semantic Scholar API"""
        if self.cache:
            cached = self.cache.get_search_results(query, "semantic_scholar")
            if cached:
                return self._nodes_from_cached_results(cached, "paper", depth, parent_node, query)
        
        nodes = []
        
        try:
            headers = {}
            if self.config.SEMANTIC_SCHOLAR_API_KEY:
                headers['x-api-key'] = self.config.SEMANTIC_SCHOLAR_API_KEY
            
            url = f"https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': self.config.MAX_SEMANTIC_SCHOLAR_PER_QUERY,
                'fields': 'title,abstract,authors,year,citationCount,url,referenceCount'
            }
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for paper in data.get('data', []):
                    paper_url = paper.get('url', f"https://semanticscholar.org/paper/{paper.get('paperId', '')}")
                    
                    if paper_url in self.processed_urls:
                        continue
                    
                    self.processed_urls.add(paper_url)
                    
                    title = paper.get('title', '')
                    abstract = paper.get('abstract', '')
                    
                    if not abstract:
                        continue
                    
                    text = f"Title: {title}\n\nAbstract: {abstract}"
                    
                    authors = [a.get('name', '') for a in paper.get('authors', [])]
                    
                    node = self._create_node(
                        text=text,
                        url=paper_url,
                        node_type="paper",
                        depth=depth,
                        parent_node=parent_node,
                        query=query,
                        metadata={
                            'title': title,
                            'authors': authors,
                            'publication_date': str(paper.get('year', '')),
                            'citations': paper.get('citationCount', 0),
                            'references': paper.get('referenceCount', 0)
                        }
                    )
                    
                    node.scientific_citations = paper.get('citationCount', 0)
                    nodes.append(node)
                    
                    results.append({
                        'title': title,
                        'abstract': abstract,
                        'url': paper_url,
                        'authors': authors
                    })
                
                if self.cache:
                    self.cache.set_search_results(query, "semantic_scholar", results)
        
        except Exception as e:
            logger.error(f"Semantic Scholar search error for '{query}': {e}")
        
        return nodes
    
    def _search_crossref(self, query: str, depth: int, parent_node: Optional[Node]) -> List[Node]:
        """Search Crossref API"""
        if self.cache:
            cached = self.cache.get_search_results(query, "crossref")
            if cached:
                return self._nodes_from_cached_results(cached, "paper", depth, parent_node, query)
        
        nodes = []
        
        try:
            results_data = []
            works = self.crossref_works.query(query).select('title', 'abstract', 'author', 'published', 'DOI', 'URL')
            
            for i, item in enumerate(works):
                if i >= self.config.MAX_CROSSREF_PER_QUERY:
                    break
                
                url = item.get('URL', f"https://doi.org/{item.get('DOI', '')}")
                
                if url in self.processed_urls:
                    continue
                
                self.processed_urls.add(url)
                
                title = item.get('title', [''])[0] if 'title' in item else ''
                abstract = item.get('abstract', '')
                
                if not abstract and not title:
                    continue
                
                text = f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"
                
                authors = []
                if 'author' in item:
                    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item['author']]
                
                pub_date = None
                if 'published' in item:
                    date_parts = item['published'].get('date-parts', [[]])[0]
                    if date_parts:
                        pub_date = f"{date_parts[0]}"
                
                node = self._create_node(
                    text=text,
                    url=url,
                    node_type="paper",
                    depth=depth,
                    parent_node=parent_node,
                    query=query,
                    metadata={
                        'title': title,
                        'authors': authors,
                        'publication_date': pub_date
                    }
                )
                
                nodes.append(node)
                results_data.append({
                    'title': title,
                    'abstract': abstract,
                    'url': url,
                    'authors': authors
                })
            
            if self.cache:
                self.cache.set_search_results(query, "crossref", results_data)
        
        except Exception as e:
            logger.error(f"Crossref search error for '{query}': {e}")
        
        return nodes
    
    def _search_pubmed(self, query: str, depth: int, parent_node: Optional[Node]) -> List[Node]:
        """Search PubMed via Entrez API"""
        if self.cache:
            cached = self.cache.get_search_results(query, "pubmed")
            if cached:
                return self._nodes_from_cached_results(cached, "paper", depth, parent_node, query)
        
        nodes = []
        
        try:
            from Bio import Entrez
            Entrez.email = "FeklinDN@gmail.com"
            
            handle = Entrez.esearch(db="pubmed", term=query, retmax=self.config.MAX_PUBMED_PER_QUERY)
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            
            if not id_list:
                return nodes
            
            handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            results_data = []
            
            for article in records['PubmedArticle']:
                medline = article['MedlineCitation']
                article_data = medline['Article']
                
                pmid = str(medline['PMID'])
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                
                if url in self.processed_urls:
                    continue
                
                self.processed_urls.add(url)
                
                title = article_data.get('ArticleTitle', '')
                abstract = article_data.get('Abstract', {}).get('AbstractText', [''])[0] if 'Abstract' in article_data else ''
                
                if not abstract and not title:
                    continue
                
                text = f"Title: {title}\n\nAbstract: {abstract}"
                
                authors = []
                if 'AuthorList' in article_data:
                    authors = [f"{a.get('ForeName', '')} {a.get('LastName', '')}".strip() for a in article_data['AuthorList']]
                
                node = self._create_node(
                    text=text,
                    url=url,
                    node_type="paper",
                    depth=depth,
                    parent_node=parent_node,
                    query=query,
                    metadata={
                        'title': title,
                        'authors': authors
                    }
                )
                
                nodes.append(node)
                results_data.append({
                    'title': title,
                    'abstract': abstract,
                    'url': url,
                    'authors': authors
                })
            
            if self.cache:
                self.cache.set_search_results(query, "pubmed", results_data)
        
        except Exception as e:
            logger.error(f"PubMed search error for '{query}': {e}")
        
        return nodes
    
    def _search_google_scholar(self, query: str, depth: int, parent_node: Optional[Node]) -> List[Node]:
        """Google Scholar через scholarly (с fallback на web)"""
        nodes = []
        try:
            search_query = scholarly.search_pubs(query)
            for i, pub in enumerate(search_query):
                if i >= self.config.MAX_SCHOLAR_PER_QUERY:
                    break

                bib = pub.get('bib', {})
                title = bib.get('title', '')
                abstract = pub.get('abstract', '') or bib.get('abstract', '')
                url = pub.get('eprint', pub.get('pub_url', ''))

                if not url or url in self.processed_urls:
                    continue

                self.processed_urls.add(url)

                text = f"Title: {title}\nAbstract: {abstract}"
                node = self._create_node(
                    text=text,
                    url=url,
                    node_type="paper",
                    depth=depth,
                    parent_node=parent_node,
                    query=query,
                    metadata={'title': title}
                )
                nodes.append(node)

        except Exception as e:
            logger.warning(f"Google Scholar failed: {e}. Falling back to web search.")
            # Fallback — обычный DDGS с site:scholar.google.com
            with DDGS() as ddgs:
                results = ddgs.text(f"{query} site:scholar.google.com", max_results=10)
                for r in results:
                    text = f"Title: {r['title']}\n{r['body']}"
                    node = self._create_node(text, r['href'], "paper", depth, parent_node, query)
                    nodes.append(node)

        return nodes
    
    def _search_patents(self, query: str, depth: int, parent_node: Optional[Node]) -> List[Node]:
        """Search patents via web scraping"""
        if self.cache:
            cached = self.cache.get_search_results(query, "patents")
            if cached:
                return self._nodes_from_cached_results(cached, "patent", depth, parent_node, query)
        
        nodes = []
        
        # Use DuckDuckGo with "site:patents.google.com"
        patent_query = f"{query} site:patents.google.com"
        
        try:
            with DDGS() as ddgs:
                results = ddgs.text(patent_query, max_results=self.config.MAX_PATENTS_PER_QUERY)
                results_data = []
                
                for r in results:
                    url = r["href"]
                    
                    if url in self.processed_urls:
                        continue
                    
                    self.processed_urls.add(url)
                    
                    text = f"Title: {r['title']}\n\n{r['body']}"
                    
                    node = self._create_node(
                        text=text,
                        url=url,
                        node_type="patent",
                        depth=depth,
                        parent_node=parent_node,
                        query=query,
                        metadata={'title': r['title']}
                    )
                    
                    nodes.append(node)
                    results_data.append({
                        'title': r['title'],
                        'body': r['body'],
                        'url': url
                    })
                
                if self.cache:
                    self.cache.set_search_results(query, "patents", results_data)
        
        except Exception as e:
            logger.error(f"Patent search error for '{query}': {e}")
        
        return nodes
    
    def _search_github(self, query: str, depth: int, parent_node: Optional[Node]) -> List[Node]:
        """Search GitHub repositories"""
        if self.cache:
            cached = self.cache.get_search_results(query, "github")
            if cached:
                return self._nodes_from_cached_results(cached, "code", depth, parent_node, query)
        
        nodes = []
        
        # Use DuckDuckGo with "site:github.com"
        github_query = f"{query} site:github.com"
        
        try:
            with DDGS() as ddgs:
                results = ddgs.text(github_query, max_results=self.config.MAX_GITHUB_PER_QUERY)
                results_data = []
                
                for r in results:
                    url = r["href"]
                    
                    if url in self.processed_urls or '/issues/' in url or '/pull/' in url:
                        continue
                    
                    self.processed_urls.add(url)
                    
                    text = f"Title: {r['title']}\n\n{r['body']}"
                    
                    node = self._create_node(
                        text=text,
                        url=url,
                        node_type="code",
                        depth=depth,
                        parent_node=parent_node,
                        query=query,
                        metadata={'title': r['title']}
                    )
                    
                    nodes.append(node)
                    results_data.append({
                        'title': r['title'],
                        'body': r['body'],
                        'url': url
                    })
                
                if self.cache:
                    self.cache.set_search_results(query, "github", results_data)
        
        except Exception as e:
            logger.error(f"GitHub search error for '{query}': {e}")
        
        return nodes
    
    def _search_web(self, query: str, depth: int, parent_node: Optional[Node]) -> List[Node]:
        """Search general web"""
        if self.cache:
            cached = self.cache.get_search_results(query, "web")
            if cached:
                return self._nodes_from_cached_results(cached, "web", depth, parent_node, query)
        
        nodes = []
        
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=self.config.MAX_WEB_PER_QUERY)
                results_data = []
                
                for r in results:
                    url = r["href"]
                    
                    if url in self.processed_urls:
                        continue
                    
                    self.processed_urls.add(url)
                    
                    node_type = self._classify_url(url)
                    
                    text = f"Title: {r['title']}\n\n{r['body']}"
                    
                    node = self._create_node(
                        text=text,
                        url=url,
                        node_type=node_type,
                        depth=depth,
                        parent_node=parent_node,
                        query=query,
                        metadata={'title': r['title']}
                    )
                    
                    nodes.append(node)
                    results_data.append({
                        'title': r['title'],
                        'body': r['body'],
                        'url': url
                    })
                
                if self.cache:
                    self.cache.set_search_results(query, "web", results_data)
        
        except Exception as e:
            logger.error(f"Web search error for '{query}': {e}")
        
        return nodes
    
    def _classify_url(self, url: str) -> str:
        """Classify node type based on URL"""
        url_lower = url.lower()
        
        if any(x in url_lower for x in ['arxiv.org', 'doi.org', 'pubmed', 'semanticscholar']):
            return 'paper'
        elif any(x in url_lower for x in ['patent', 'uspto', 'espacenet']):
            return 'patent'
        elif any(x in url_lower for x in ['github.com', 'gitlab']):
            return 'code'
        elif any(x in url_lower for x in ['reddit.com', 'hackernews', 'forum']):
            return 'forum'
        elif any(x in url_lower for x in ['crunchbase', 'techcrunch', 'funding']):
            return 'startup'
        elif any(x in url_lower for x in ['youtube', 'vimeo']):
            return 'media'
        elif any(x in url_lower for x in ['.gov', 'europa.eu']):
            return 'regulatory'
        elif any(x in url_lower for x in ['news', 'forbes', 'wired']):
            return 'media'
        else:
            return 'web'
    
    def _nodes_from_cached_results(self, cached: list, source_type: str, depth: int, parent_node: Optional[Node], query: str) -> List[Node]:
        """Recreate nodes from cached results. Safe against missing 'abstract' / 'body' keys."""
        nodes = []
        
        for item in cached:
            url = item.get('url', '')
            
            if url in self.processed_urls:
                continue
            
            self.processed_urls.add(url)
            
            title   = item.get('title', '')
            body    = item.get('abstract', '') or item.get('body', '') or ''
            text    = f"Title: {title}\n\n{body}".strip()
            
            if not text:
                continue
            
            node_type = source_type if source_type != "web" else self._classify_url(url)
            
            node = self._create_node(
                text=text,
                url=url,
                node_type=node_type,
                depth=depth,
                parent_node=parent_node,
                query=query,
                metadata=item
            )
            
            nodes.append(node)
        
        return nodes
    
    # ================================================================
    # SOURCE 9: ResearchGate (web scraping + link verification)
    # ================================================================

    def _search_researchgate(self, query: str, depth: int,
                              parent_node: Optional["Node"]) -> List["Node"]:
        """
        Поиск публикаций на ResearchGate через DuckDuckGo (site:researchgate.net).
        Найденные URL верифицируются HEAD-запросом перед созданием узла.
        """
        if self.cache:
            cached = self.cache.get_search_results(query, "researchgate")
            if cached:
                return self._nodes_from_cached_results(cached, "paper", depth, parent_node, query)

        nodes = []
        rg_query = f"{query} site:researchgate.net/publication"

        try:
            with DDGS() as ddgs:
                results = ddgs.text(rg_query, max_results=self.config.MAX_RESEARCHGATE_PER_QUERY)
            results_data = []

            for r in results:
                url = r["href"]

                if url in self.processed_urls:
                    continue

                # Верификация ссылки: проверяем, что страница реально существует
                if not self._verify_url(url):
                    logger.debug(f"ResearchGate URL не прошёл верификацию: {url}")
                    continue

                self.processed_urls.add(url)
                text = f"Title: {r['title']}\n\n{r['body']}"
                node = self._create_node(text, url, "paper", depth, parent_node, query,
                                         metadata={"title": r["title"]})
                nodes.append(node)
                results_data.append({"title": r["title"], "body": r["body"], "url": url})

            if self.cache and results_data:
                self.cache.set_search_results(query, "researchgate", results_data)

        except Exception as e:
            logger.error(f"ResearchGate search error for '{query}': {e}")

        return nodes

    def _verify_url(self, url: str, timeout: int = 5) -> bool:
        """
        Проверяет доступность URL через HEAD-запрос.
        Возвращает True, если статус 200-399.
        """
        try:
            resp = requests.head(url, timeout=timeout, allow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0"})
            return resp.status_code < 400
        except Exception:
            return False

    # ================================================================
    # SOURCE 10: Форумы — Reddit, HackerNews, StackExchange
    # ================================================================

    def _search_forums(self, query: str, depth: int,
                       parent_node: Optional["Node"]) -> List["Node"]:
        """
        Ищет обсуждения на форумах через DuckDuckGo.
        Если задан REDDIT_CLIENT_ID — использует Reddit API напрямую.
        Заполняет forum_post_count и forum_sentiment_raw узла.
        """
        if self.cache:
            cached = self.cache.get_search_results(query, "forums")
            if cached:
                return self._nodes_from_cached_results(cached, "forum", depth, parent_node, query)

        nodes      = []
        forum_sites = [
            "site:reddit.com",
            "site:news.ycombinator.com",
            "site:stackoverflow.com",
            "site:stackexchange.com",
            "site:researchgate.net/post",
        ]
        results_data = []

        for site_filter in forum_sites:
            forum_query = f"{query} {site_filter}"
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(forum_query,
                                        max_results=self.config.MAX_FORUM_PER_QUERY // len(forum_sites) + 1)
                for r in results:
                    url = r["href"]
                    if url in self.processed_urls:
                        continue
                    self.processed_urls.add(url)
                    text = f"Title: {r['title']}\n\n{r['body']}"
                    node = self._create_node(text, url, "forum", depth, parent_node, query,
                                             metadata={"title": r["title"]})
                    # Подсчёт постов и сырой sentiment
                    node.forum_post_count += 1
                    nodes.append(node)
                    results_data.append({"title": r["title"], "body": r["body"], "url": url})
            except Exception as e:
                logger.warning(f"Forum search ({site_filter}) error for '{query}': {e}")

        # Опциональный прямой доступ к Reddit API
        if self.config.REDDIT_CLIENT_ID and self.config.REDDIT_CLIENT_SECRET:
            reddit_nodes = self._search_reddit_api(query, depth, parent_node)
            nodes.extend(reddit_nodes)

        if self.cache and results_data:
            self.cache.set_search_results(query, "forums", results_data)

        return nodes

    def _search_reddit_api(self, query: str, depth: int,
                           parent_node: Optional["Node"]) -> List["Node"]:
        """
        Поиск через Reddit API (PRAW-совместимый OAuth).
        Активируется только при наличии REDDIT_CLIENT_ID и REDDIT_CLIENT_SECRET.
        """
        nodes = []
        try:
            token_resp = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(self.config.REDDIT_CLIENT_ID, self.config.REDDIT_CLIENT_SECRET),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": "GraphForecasterV6/1.0"},
                timeout=10,
            )
            token = token_resp.json().get("access_token", "")
            if not token:
                return nodes

            headers = {"Authorization": f"Bearer {token}", "User-Agent": "GraphForecasterV6/1.0"}
            search_resp = requests.get(
                "https://oauth.reddit.com/search",
                params={"q": query, "limit": 25, "sort": "relevance", "type": "link"},
                headers=headers,
                timeout=10,
            )
            for post in search_resp.json().get("data", {}).get("children", []):
                d   = post["data"]
                url = f"https://reddit.com{d.get('permalink', '')}"
                if url in self.processed_urls:
                    continue
                self.processed_urls.add(url)
                text = f"Title: {d.get('title','')}\n\n{d.get('selftext','')}"
                node = self._create_node(text, url, "forum", depth, parent_node, query,
                                         metadata={"title": d.get("title",""),
                                                   "upvotes": d.get("score", 0)})
                node.forum_post_count = d.get("num_comments", 0)
                nodes.append(node)
        except Exception as e:
            logger.warning(f"Reddit API error for '{query}': {e}")
        return nodes

    # ================================================================
    # SOURCE 11: Инвестиционные и финансовые данные (заглушки)
    # ================================================================

    def _fetch_investment_data(self, query: str, depth: int,
                               parent_node: Optional["Node"]) -> List["Node"]:
        """
        Агрегатор заглушек для финансовых баз данных.
        При наличии API-ключей — делегирует конкретным коннекторам.
        Без ключей — возвращает пустой список с предупреждением.
        """
        nodes: List["Node"] = []

        if self.config.CRUNCHBASE_API_KEY:
            nodes.extend(self._fetch_crunchbase(query, depth, parent_node))
        else:
            logger.debug("Crunchbase API key не задан — источник пропущен")

        if self.config.PITCHBOOK_API_KEY:
            nodes.extend(self._fetch_pitchbook(query, depth, parent_node))
        else:
            logger.debug("PitchBook API key не задан — источник пропущен")

        if self.config.DEALROOM_API_KEY:
            nodes.extend(self._fetch_dealroom(query, depth, parent_node))
        else:
            logger.debug("Dealroom API key не задан — источник пропущен")

        # Fallback: публичный поиск инвестиционных новостей через DuckDuckGo
        if not nodes:
            nodes.extend(self._fetch_investment_news_web(query, depth, parent_node))

        return nodes

    def _fetch_crunchbase(self, query: str, depth: int,
                          parent_node: Optional["Node"]) -> List["Node"]:
        """
        ЗАГЛУШКА: Crunchbase API v4.
        Документация: https://data.crunchbase.com/docs
        Подключить: задать CRUNCHBASE_API_KEY и реализовать запрос к
        https://api.crunchbase.com/api/v4/searches/organizations
        
        Формат ответа ожидается:
          { "entities": [{ "properties": { "short_description": ...,
                                           "total_funding_usd": ...,
                                           "num_funding_rounds": ...,
                                           "investor_identifiers": [...] } }] }
        """
        logger.info(f"[STUB] Crunchbase: запрос для '{query}'")
        # TODO: реализовать реальный запрос при наличии ключа
        # Пример:
        # resp = requests.post(
        #     "https://api.crunchbase.com/api/v4/searches/organizations",
        #     json={"field_ids": ["short_description","total_funding_usd",
        #                         "num_funding_rounds","investor_identifiers"],
        #           "query": [{"type":"predicate","field_id":"facet_ids",
        #                      "operator_id":"includes","values":["company"]}],
        #           "limit": self.config.MAX_INVESTMENT_PER_QUERY},
        #     params={"user_key": self.config.CRUNCHBASE_API_KEY},
        # )
        # for entity in resp.json().get("entities", []):
        #     props = entity["properties"]
        #     node = self._create_node(props["short_description"], ...)
        #     node.investment_total_usd = props.get("total_funding_usd", 0)
        #     ...
        return []

    def _fetch_pitchbook(self, query: str, depth: int,
                         parent_node: Optional["Node"]) -> List["Node"]:
        """
        ЗАГЛУШКА: PitchBook Data API.
        Документация: https://pitchbook.com/news/articles/pitchbook-api
        Подключить: задать PITCHBOOK_API_KEY, использовать Basic Auth.
        Endpoint: https://api.pitchbook.com/entities?query=...
        
        Ожидаемые поля: totalRaised, lastRoundSize, leadInvestors, description.
        """
        logger.info(f"[STUB] PitchBook: запрос для '{query}'")
        # TODO: реализовать при наличии ключа
        return []

    def _fetch_dealroom(self, query: str, depth: int,
                        parent_node: Optional["Node"]) -> List["Node"]:
        """
        ЗАГЛУШКА: Dealroom.co API.
        Документация: https://dealroom.co/api
        Подключить: задать DEALROOM_API_KEY.
        Endpoint: https://api.dealroom.co/api/v1/companies?q=...
        
        Ожидаемые поля: total_funding, last_round, investors, tagline.
        """
        logger.info(f"[STUB] Dealroom: запрос для '{query}'")
        # TODO: реализовать при наличии ключа
        return []

    def _fetch_investment_news_web(self, query: str, depth: int,
                                   parent_node: Optional["Node"]) -> List["Node"]:
        """
        Fallback: поиск инвестиционных новостей через DuckDuckGo.
        Ищет в Crunchbase News, TechCrunch, VentureBeat.
        """
        if self.cache:
            cached = self.cache.get_search_results(query, "investment_web")
            if cached:
                return self._nodes_from_cached_results(cached, "startup", depth, parent_node, query)

        nodes = []
        invest_query = (
            f"{query} funding investment round "
            f"(site:techcrunch.com OR site:venturebeat.com OR site:crunchbase.com)"
        )
        results_data = []
        try:
            with DDGS() as ddgs:
                results = ddgs.text(invest_query, max_results=self.config.MAX_INVESTMENT_PER_QUERY)
            for r in results:
                url = r["href"]
                if url in self.processed_urls:
                    continue
                self.processed_urls.add(url)
                text = f"Title: {r['title']}\n\n{r['body']}"
                node = self._create_node(text, url, "startup", depth, parent_node, query,
                                         metadata={"title": r["title"]})
                nodes.append(node)
                results_data.append({"title": r["title"], "body": r["body"], "url": url})

            if self.cache and results_data:
                self.cache.set_search_results(query, "investment_web", results_data)
        except Exception as e:
            logger.error(f"Investment web search error for '{query}': {e}")
        return nodes

    # ================================================================
    # SOURCE 12: Память модели + верификация ссылок
    # ================================================================

    def _search_model_memory(self, query: str, depth: int,
                             parent_node: Optional["Node"]) -> List["Node"]:
        """
        Запрашивает LLM о ключевых работах и технологиях по теме из «памяти»
        (знаний, встроенных в модель при обучении).
        Каждая возвращённая ссылка проходит верификацию (_verify_url).
        Непроверяемые ссылки (нет HTTP-доступа) сохраняются как узлы с пометкой
        'unverified_memory_reference', но не удаляются — они могут быть ценными.
        """
        if self.cache:
            cached = self.cache.get_search_results(query, "model_memory")
            if cached:
                return self._nodes_from_cached_results(cached, "paper", depth, parent_node, query)

        prompt = f"""
        From your training knowledge, list up to 10 key papers, technologies,
        or resources relevant to: "{query}"

        For each item provide:
        - title
        - authors (if known)
        - year (if known)
        - likely URL (arXiv, DOI, GitHub, etc.) or empty string if unknown
        - brief description (2-3 sentences)

        Return JSON array:
        [
          {{
            "title": "...",
            "authors": ["..."],
            "year": "...",
            "url": "...",
            "description": "..."
          }},
          ...
        ]
        Only include items you are reasonably confident exist.
        """

        response = self._call_llm(prompt, temperature=0.1, response_format="json")
        nodes = []
        results_data = []

        try:
            items = json.loads(response)
            if not isinstance(items, list):
                items = []
        except Exception:
            items = []

        for item in items:
            title       = item.get("title", "").strip()
            description = item.get("description", "").strip()
            url         = item.get("url", "").strip()
            authors     = item.get("authors", [])
            year        = str(item.get("year", ""))

            if not title and not description:
                continue

            text = f"Title: {title}\n\nDescription: {description}"

            # Верификация ссылки
            verified = False
            if url:
                verified = self._verify_url(url)
                if not verified:
                    logger.debug(f"Memory reference URL не верифицирован: {url}")
                    # Не выбрасываем — оставляем с пометкой
                    url = url  # сохраняем для трассировки

            if url in self.processed_urls:
                continue
            if url:
                self.processed_urls.add(url)

            node = self._create_node(
                text=text,
                url=url or f"memory://{hashlib.md5(title.encode()).hexdigest()[:12]}",
                node_type="paper",
                depth=depth,
                parent_node=parent_node,
                query=query,
                metadata={
                    "title": title,
                    "authors": authors,
                    "publication_date": year,
                    "memory_verified": verified,
                    "source": "model_memory",
                }
            )

            if not verified:
                # Добавляем в description пометку об непроверенности
                node.description = f"[UNVERIFIED MEMORY REF] {node.description}"

            nodes.append(node)
            results_data.append({
                "title": title,
                "body": description,
                "url": url,
                "verified": verified,
            })

        if self.cache and results_data:
            self.cache.set_search_results(query, "model_memory", results_data)

        logger.info(f"Model memory: {len(nodes)} references found for '{query[:60]}' "
                    f"({sum(1 for n in nodes if 'UNVERIFIED' not in n.description)} verified)")
        return nodes

    # ================================================================
    # SOCIAL PERCEPTION — sentiment-модели на рецензиях и художественных текстах
    # ================================================================

    def _load_sentiment_pipelines(self):
        """
        Ленивая загрузка sentiment-моделей.
        Вызывается только при первом обращении к score_social_perception.
        """
        if not TRANSFORMERS_AVAILABLE:
            return

        from transformers import pipeline

        if self._sentiment_review_pipe is None:
            try:
                logger.info(f"Loading review sentiment model: {self.config.SENTIMENT_REVIEW_MODEL}")
                self._sentiment_review_pipe = pipeline(
                    "sentiment-analysis",
                    model=self.config.SENTIMENT_REVIEW_MODEL,
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                logger.warning(f"Не удалось загрузить review-модель: {e}")
                self._sentiment_review_pipe = None

        if self._sentiment_fiction_pipe is None:
            try:
                logger.info(f"Loading fiction/emotion model: {self.config.SENTIMENT_FICTION_MODEL}")
                self._sentiment_fiction_pipe = pipeline(
                    "text-classification",
                    model=self.config.SENTIMENT_FICTION_MODEL,
                    truncation=True,
                    max_length=512,
                    top_k=None,
                )
            except Exception as e:
                logger.warning(f"Не удалось загрузить fiction-модель: {e}")
                self._sentiment_fiction_pipe = None

    def _score_review_sentiment(self, text: str) -> float:
        """
        Оценивает текст через модель, обученную на рецензиях (nlptown).
        Возвращает нормализованный скор 0.0–1.0 (1 = максимально позитивно).
        Модель возвращает метки типа '5 stars' / '1 star'.
        """
        if not TRANSFORMERS_AVAILABLE or self._sentiment_review_pipe is None:
            return 0.0
        try:
            result = self._sentiment_review_pipe(text[:512])[0]
            label: str = result.get("label", "3 stars")
            # Метки: '1 star' … '5 stars'
            stars = int(label.split()[0]) if label[0].isdigit() else 3
            return (stars - 1) / 4.0   # нормализация 1-5 → 0.0-1.0
        except Exception as e:
            logger.debug(f"Review sentiment error: {e}")
            return 0.0

    def _score_fiction_sentiment(self, text: str) -> float:
        """
        Оценивает текст через модель эмоций (j-hartmann/emotion).
        Возвращает «позитивность» 0.0–1.0 как взвешенную сумму:
            joy + surprise → позитив
            fear + disgust + sadness + anger → негатив
        """
        if not TRANSFORMERS_AVAILABLE or self._sentiment_fiction_pipe is None:
            return 0.0
        try:
            results = self._sentiment_fiction_pipe(text[:512])[0]
            pos_labels = {"joy", "surprise", "love"}
            neg_labels = {"fear", "disgust", "sadness", "anger"}
            pos_score = sum(r["score"] for r in results if r["label"].lower() in pos_labels)
            neg_score = sum(r["score"] for r in results if r["label"].lower() in neg_labels)
            total = pos_score + neg_score
            return pos_score / total if total > 0 else 0.5
        except Exception as e:
            logger.debug(f"Fiction sentiment error: {e}")
            return 0.0

    def score_social_perception(self, nodes: Optional[List["Node"]] = None) -> None:
        """
        Считает social_perception_score для каждого узла на основе:
          1. review_sentiment  — восприятие широкой аудиторией (модель на рецензиях)
          2. fiction_sentiment — нарративное восприятие (модель на художественных текстах)
          3. forum_sentiment   — тональность форумных обсуждений (из forum_sentiment_raw)

        Веса: review 0.35, fiction 0.30, forum 0.35.
        Обновляет node.social_score и node.social_perception_score.
        """
        self._load_sentiment_pipelines()

        target_nodes = list(self.nodes.values()) if nodes is None else nodes
        logger.info(f"Scoring social perception for {len(target_nodes)} nodes...")

        for node in tqdm(target_nodes, desc="Social perception"):
            text = node.full_text[:1500] if node.full_text else node.description

            # 1. Review sentiment
            review_score   = self._score_review_sentiment(text)
            node.sentiment_review_score = review_score

            # 2. Fiction / emotion sentiment
            fiction_score  = self._score_fiction_sentiment(text)
            node.sentiment_fiction_score = fiction_score

            # 3. Forum sentiment (среднее из сырых оценок, если есть)
            if node.forum_sentiment_raw:
                forum_score = float(np.mean(node.forum_sentiment_raw))
            else:
                forum_score = 0.5   # нейтральный дефолт при отсутствии данных
            node.sentiment_forum_score = forum_score

            # Агрегируем
            node.social_perception_score = (
                0.35 * review_score +
                0.30 * fiction_score +
                0.35 * forum_score
            )

            # Обновляем social_score (используется в readiness / forecast)
            node.social_score = node.social_perception_score * 10.0   # масштаб 0-10

    # ================================================================
    # DOMAIN EXTRACTION & CROSS-DOMAIN DISCOVERY (CORRECTED)
    # ================================================================
    
    def _extract_domains_from_nodes(self, nodes: List[Node]):
        """
        Extract mentioned domains and cited works from nodes using LLM
        Process in batches for efficiency
        """
        unprocessed = [n for n in nodes if not n.domains_extracted]
        
        if not unprocessed:
            return
        
        logger.info(f"Extracting domains from {len(unprocessed)} nodes")
        
        # Process in batches
        batch_size = self.config.CROSS_DOMAIN_BATCH_SIZE
        
        for i in range(0, len(unprocessed), batch_size):
            batch = unprocessed[i:i + batch_size]
            
            # Prepare batch text
            batch_texts = []
            for idx, node in enumerate(batch):
                text_sample = node.full_text[:2000] if node.full_text else node.description
                batch_texts.append(f"[Document {idx}]\n{text_sample}\n")
            
            combined_text = "\n".join(batch_texts)
            
            # LLM extraction
            prompt = f"""
            Analyze these {len(batch)} technical documents and extract:
            
            1. **Technical domains** mentioned (e.g., "laser physics", "semiconductor manufacturing", "biomedical imaging")
            2. **Technologies referenced** (specific techniques, materials, devices)
            3. **Related fields** cited or discussed
            
            {combined_text}
            
            Return JSON array with one object per document:
            [
              {{
                "doc_id": 0,
                "mentioned_domains": ["domain1", "domain2", ...],
                "cited_technologies": ["tech1", "tech2", ...],
                "related_fields": ["field1", "field2", ...]
              }},
              ...
            ]
            
            Be specific and technical. Extract ALL domains mentioned, even briefly.
            """
            
            response = self._call_llm(prompt, temperature=0.1, max_tokens=4000)
            
            try:
                extractions = json.loads(response)
                
                for extraction in extractions:
                    doc_id = extraction.get('doc_id', 0)
                    if doc_id >= len(batch):
                        continue
                    
                    node = batch[doc_id]
                    
                    # Update node
                    domains = extraction.get('mentioned_domains', [])
                    technologies = extraction.get('cited_technologies', [])
                    fields = extraction.get('related_fields', [])
                    
                    node.mentioned_domains = domains
                    node.cited_works = technologies
                    node.key_concepts.extend(fields)
                    
                    # Add to global domain tracking
                    self.discovered_domains.update(domains)
                    self.discovered_domains.update(fields)
                    
                    node.domains_extracted = True
            
            except Exception as e:
                logger.error(f"Domain extraction error: {e}")
                # Mark as processed anyway to avoid retry loops
                for node in batch:
                    node.domains_extracted = True
            
            self.rate_limiter.wait_if_needed()
        
        logger.info(f"Total discovered domains: {len(self.discovered_domains)}")
    
    def discover_cross_domain_analogies(self, max_analogies_per_domain: int = 10):
        """
        CORRECTED: Discover cross-domain solutions based on extracted domains
        """
        logger.info("STAGE 3: Cross-domain analogy discovery (CORRECTED)")
        
        if not self.discovered_domains:
            logger.warning("No domains discovered yet. Run ingest_all_sources first.")
            return
        
        logger.info(f"Discovered domains: {sorted(self.discovered_domains)[:20]}")
        
        # Group limitations by category
        limitations = self.problem_tree.get('known_limitations', [])
        
        limitations_by_category = defaultdict(list)
        for lim in limitations:
            cat = lim.get('category', 'general')
            limitations_by_category[cat].append(lim['limitation'])
        
        # For each category, find analogies in discovered domains
        for category, lims in limitations_by_category.items():
            logger.info(f"\nCategory: {category}")
            logger.info(f"Limitations: {lims[:2]}...")
            
            # Generate cross-domain queries based on ACTUAL discovered domains
            analogy_queries = self._generate_domain_specific_queries(
                category=category,
                limitations=lims[:3],
                discovered_domains=list(self.discovered_domains)
            )
            
            logger.info(f"Generated {len(analogy_queries)} cross-domain queries")
            
            # Execute queries
            analogy_nodes = self._execute_multi_source_queries(
                analogy_queries[:max_analogies_per_domain],
                depth=0
            )
            
            # Tag nodes
            for node in analogy_nodes:
                node.node_type = "analogy"
                for lim in lims:
                    node.solves_limitations.append(lim)
            
            logger.info(f"Found {len(analogy_nodes)} cross-domain analogies")
    
    def _generate_domain_specific_queries(self, category: str, limitations: List[str], discovered_domains: List[str]) -> List[str]:
        """
        Generate queries that search for solutions IN the discovered domains
        """
        # Limit to most relevant domains (top 30)
        domain_sample = discovered_domains[:30]
        
        prompt = f"""
        We need to solve these problems:
        Category: {category}
        Specific limitations: {limitations}
        
        We've discovered these technical domains are related to our problem:
        {domain_sample}
        
        Generate 15 specific search queries to find solutions from these ACTUAL domains.
        
        Format: "[Domain] + [specific technique/approach for solving the problem]"
        
        Example:
        - "semiconductor lithography beam steering techniques"
        - "biomedical imaging parallel acquisition methods"
        - "telecommunications signal processing cost reduction"
        
        Return JSON array of query strings.
        Focus on CONCRETE techniques from the discovered domains that could solve the stated limitations.
        """
        
        response = self._call_llm(prompt, temperature=0.6, max_tokens=2000)
        
        try:
            queries = json.loads(response)
            return queries if isinstance(queries, list) else []
        except:
            return []
    
    # ================================================================
    # REST OF THE CODE (Same as before but using corrected methods)
    # ================================================================
    
    def _generate_comprehensive_queries(self, topic: str, query_type: str) -> List[str]:
        """Generate diverse queries"""
        prompt = f"""
        Topic: {topic}
        Query Type: {query_type}
        
        Generate 20 diverse search queries covering:
        1. Academic research
        2. Patents and IP
        3. Commercial products
        4. Technical forums
        5. Government/regulatory
        6. Related technologies
        7. Historical development
        8. Alternative approaches
        9. Recent news
        10. Industry reports
        
        Return JSON array of query strings.
        """
        
        response = self._call_llm(prompt, temperature=0.6)
        
        try:
            queries = json.loads(response)
            return queries if isinstance(queries, list) else [topic]
        except:
            return [topic]
    
    def _generate_expansion_queries(self, node: Node) -> List[str]:
        """Generate queries to expand from node"""
        prompt = f"""
        Given this technology:
        Title: {node.title or node.description[:100]}
        Advantages: {node.advantages[:3]}
        Limitations: {node.limitations[:3]}
        Mentioned domains: {node.mentioned_domains[:5]}
        
        Generate 8 search queries to find:
        1. Solutions to limitations
        2. Technologies this enables
        3. Prerequisites
        4. Competing approaches
        5. Recent developments in mentioned domains
        
        Return JSON array.
        """
        
        response = self._call_llm(prompt, temperature=0.5)
        
        try:
            queries = json.loads(response)
            return queries[:8] if isinstance(queries, list) else []
        except:
            return []
    
    def _create_node(self, text: str, url: str, node_type: str, depth: int, parent_node: Optional[Node], query: str, metadata: dict = None) -> Node:
        """Create node with extraction"""
        metadata = metadata or {}
        
        embedding = self._get_embedding(text)
        extraction = self._extract_node_data(text, node_type)
        
        node_id = str(uuid.uuid4())
        
        node = Node(
            id=node_id,
            node_type=node_type,
            timestamp=datetime.utcnow().timestamp(),
            embedding=embedding,
            description=extraction.get('description', text[:200]),
            full_text=text,
            title=metadata.get('title', extraction.get('title', '')),
            advantages=extraction.get('advantages', []),
            limitations=extraction.get('limitations', []),
            key_concepts=extraction.get('key_concepts', []),
            source_urls=[url],
            authors=metadata.get('authors', []),
            publication_date=metadata.get('publication_date'),
            discovery_depth=depth,
            discovery_query=query,
            discovery_path=[parent_node.id] if parent_node else [],
            dual_use_risk=float(extraction.get('dual_use_risk', 0.0)),
            strategic_value=float(extraction.get('strategic_value', 0.0)),
            legal_risk_score=float(extraction.get('legal_risk_score', 0.0)),
            export_control_risk=float(extraction.get('export_control_risk', 0.0)),
        )
        
        # Set citations from metadata
        if 'citations' in metadata:
            node.scientific_citations = metadata['citations']
        
        self.nodes[node_id] = node
        self._update_convergence_potential(node)
        
        return node
    
    def _extract_node_data(self, text: str, node_type: str) -> dict:
        """LLM extraction"""
        prompt = f"""
        Extract from this {node_type}:
        
        {text[:12000]}
        
        Return JSON:
        {{
          "title": "concise title",
          "description": "2-3 sentence summary",
          "advantages": ["advantage1", "advantage2", ...],
          "limitations": ["limitation1", "limitation2", ...],
          "key_concepts": ["concept1", "concept2", ...],
          "maturity_level": "concept|research|prototype|product|mature",
          "dual_use_risk": <float 0-10, dual-use / военный потенциал>,
          "strategic_value": <float 0-10, стратегическая ценность>,
          "legal_risk_score": <float 0-10, правовые/регуляторные риски>,
          "export_control_risk": <float 0-10, риск экспортного контроля>
        }}
        """
        
        response = self._call_llm(prompt, temperature=0.0, response_format="json")
        
        try:
            return json.loads(response)
        except:
            return {
                'description': text[:200],
                'advantages': [],
                'limitations': [],
                'key_concepts': []
            }
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        if self.cache:
            cached = self.cache.get_embedding(text)
            if cached is not None:
                return cached
        
        embedding = self.embedder.encode([text[:8000]])[0]
        
        if self.cache:
            self.cache.set_embedding(text, embedding)
        
        return embedding
    
    def _update_convergence_potential(self, node: Node):
        """Estimate convergence potential"""
        target_embedding = self._get_embedding(self.target_problem)
        similarity = float(cosine_similarity([node.embedding], [target_embedding])[0][0])
        
        potential = (
            0.4 * similarity +
            0.3 * min(len(node.limitations) / 5, 1.0) +
            0.3 * min(len(node.advantages) / 5, 1.0)
        )
        
        node.convergence_potential = potential
    
    def _call_llm(self, prompt: str, temperature: float = 0.3, response_format: str = None, max_tokens: int = 4000) -> str:
        """Call LLM with caching"""
        if self.cache:
            cached = self.cache.get_llm_response(prompt, self.config.LLM_MODEL)
            if cached:
                return cached
        
        self.rate_limiter.wait_if_needed()
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.config.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content
            
            if self.cache:
                self.cache.set_llm_response(prompt, self.config.LLM_MODEL, result)
            
            return result
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "{}"
    
    # Additional methods (construct_all_edges, score_all_nodes, detect_convergence_clusters, etc.)
    # would follow the same pattern as in the previous version but using the corrected
    # cross-domain discovery approach
    
    # ... (include all remaining methods from previous version)

    # ================================================================
    # METRIC MERGING
    # ================================================================

    def merge_correlated_metrics(self, df: pd.DataFrame,
                                  threshold: float = 0.85) -> tuple:
        """
        Автоматически объединяет сильно коррелирующие метрики.
        Возвращает:
            - новый DataFrame
            - mapping объединённых метрик
        """
        corr = df.corr()
        used = set()
        groups = []
        mapping = {}

        for col in corr.columns:
            if col in used:
                continue
            correlated = corr.index[
                (corr[col].abs() >= threshold)
            ].tolist()
            for c in correlated:
                used.add(c)
            groups.append(correlated)

        new_df = pd.DataFrame(index=df.index)

        for group in groups:
            if len(group) == 1:
                new_df[group[0]] = df[group[0]]
                mapping[group[0]] = group[0]
            else:
                new_name = "__".join(sorted(group))
                new_df[new_name] = df[group].mean(axis=1)
                for g in group:
                    mapping[g] = new_name

        return new_df, mapping

    # ================================================================
    # STRUCTURAL DEPENDENCIES
    # ================================================================

    def compute_structural_dependencies(self, G: nx.DiGraph) -> None:
        """
        Добавляет к каждому узлу:
            structural_dependency_index
            cascade_influence
            upstream_pressure
        Также синхронизирует значения с объектами Node в self.nodes.
        """
        for node in G.nodes:
            predecessors = list(G.predecessors(node))
            successors = list(G.successors(node))

            upstream_weight = sum(
                G[p][node].get("weight", 1.0) for p in predecessors
            )
            downstream_weight = sum(
                G[node][s].get("weight", 1.0) for s in successors
            )

            G.nodes[node]["structural_dependency_index"] = upstream_weight
            G.nodes[node]["cascade_influence"] = downstream_weight
            G.nodes[node]["upstream_pressure"] = upstream_weight - downstream_weight

            # Sync back to Node objects
            if node in self.nodes:
                self.nodes[node].structural_dependency_index = upstream_weight
                self.nodes[node].cascade_influence = downstream_weight
                self.nodes[node].upstream_pressure = upstream_weight - downstream_weight

    # ================================================================
    # ACCELERATION / DRAG SCORING
    # ================================================================

    def compute_acceleration_score(self, metrics: dict) -> float:
        """
        Учитывает:
            - положительные и отрицательные значения
            - dual_use как стимул + риск
            - регуляторные барьеры как тормоз
        """
        score = 0.0

        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            score += v

        # dual-use: стимул через инвестиции и стратегию
        dual_use = metrics.get("dual_use_risk", 0.0)
        strategic = metrics.get("strategic_value", 0.0)
        dual_use_boost = dual_use * strategic * 0.1

        # регуляторное торможение
        regulatory_drag = (
            metrics.get("legal_risk_score", 0.0) +
            metrics.get("export_control_risk", 0.0)
        ) * 0.2

        return score + dual_use_boost - regulatory_drag

    # ================================================================
    # TEMPORAL ZONES
    # ================================================================

    def _infer_contained_nodes_by_clustering(self, G: nx.DiGraph) -> None:
        """
        Для узлов с is_temporal_zone=True, у которых contained_nodes пуст,
        автоматически определяет вложенные узлы через кластеризацию эмбеддингов.
        Узлы в радиусе кластера zone-узла помечаются как contained.
        """
        zone_node_ids = [
            nid for nid, data in G.nodes(data=True)
            if data.get("is_temporal_zone", False) and not data.get("contained_nodes")
        ]

        if not zone_node_ids or len(self.nodes) < 2:
            return

        # Собираем эмбеддинги всех узлов
        all_ids = list(self.nodes.keys())
        embeddings = np.array([self.nodes[nid].embedding for nid in all_ids])

        for zone_id in zone_node_ids:
            if zone_id not in self.nodes:
                continue

            zone_emb = self.nodes[zone_id].embedding.reshape(1, -1)
            sims = cosine_similarity(zone_emb, embeddings)[0]

            # Порог — выбираем узлы с sim > 0.6, исключая сам zone-узел
            threshold = 0.60
            contained = [
                all_ids[i] for i, s in enumerate(sims)
                if s >= threshold and all_ids[i] != zone_id
            ]

            if contained:
                G.nodes[zone_id]["contained_nodes"] = contained
                if zone_id in self.nodes:
                    self.nodes[zone_id].contained_nodes = contained
                logger.debug(f"Temporal zone {zone_id[:8]}… auto-assigned {len(contained)} contained nodes")

    def propagate_temporal_zone_effects(self, G: nx.DiGraph) -> None:
        """
        1. Для zone-узлов с пустым contained_nodes — автоматически определяет
           вложенные узлы через кластеризацию эмбеддингов (_infer_contained_nodes_by_clustering).
        2. Применяет zone_multiplier ко всем contained_nodes.
        3. Синхронизирует значения обратно в self.nodes.
        """
        # Шаг 1 — авто-определение contained_nodes там, где их нет
        self._infer_contained_nodes_by_clustering(G)

        # Шаг 2 — распространяем множители
        for node, data in G.nodes(data=True):
            if not data.get("is_temporal_zone", False):
                continue

            zone_multiplier = data.get("zone_multiplier", 1.0)
            contained = data.get("contained_nodes", [])

            for inner in contained:
                if inner not in G.nodes:
                    continue
                current = G.nodes[inner].get("acceleration_multiplier", 1.0)
                new_val = current * zone_multiplier
                G.nodes[inner]["acceleration_multiplier"] = new_val

                if inner in self.nodes:
                    self.nodes[inner].acceleration_multiplier = new_val

    # ================================================================
    # FORECAST SCORE PER NODE
    # ================================================================

    def compute_forecast_score(self, G: nx.DiGraph, node: str) -> float:
        """
        Итоговый прогноз с учётом:
            - метрик узла (с весами из Config.READINESS_WEIGHTS)
            - структурных зависимостей (с весами из Config.EDGE_WEIGHTS)
            - временных зон (acceleration_multiplier)
        """
        data = G.nodes.get(node, {})

        # Собираем метрики из атрибутов узла — двойной fallback
        metrics = data.get("metrics") or {}

        # Если метрики не переданы явно — строим из атрибутов Node
        if not metrics and node in self.nodes:
            n = self.nodes[node]
            metrics = {
                "scientific_score":   n.scientific_score,
                "investment_score":   n.investment_score,
                "social_score":       n.social_score,
                "maturity_score":     n.maturity_score,
                "dual_use_risk":      n.dual_use_risk,
                "strategic_value":    n.strategic_value,
                "legal_risk_score":   n.legal_risk_score,
                "export_control_risk": n.export_control_risk,
            }

        # Взвешенная readiness-компонента
        rw = self.config.READINESS_WEIGHTS
        readiness_score = (
            metrics.get("scientific_score", 0.0) * rw.get("scientific", 0.25) +
            metrics.get("investment_score", 0.0) * rw.get("investment", 0.25) +
            metrics.get("social_score",     0.0) * rw.get("social",     0.20) +
            metrics.get("maturity_score",   0.0) * rw.get("maturity",   0.15)
        )

        # Базовый acceleration score (dual-use буст + регуляторный тормоз)
        base_score = self.compute_acceleration_score(metrics) + readiness_score

        # Структурная компонента с весами из EDGE_WEIGHTS
        ew = self.config.EDGE_WEIGHTS
        semantic_w   = ew.get("semantic",   0.30)
        temporal_w   = ew.get("temporal",   0.15)
        investment_w = ew.get("investment", 0.10)

        sdi = data.get("structural_dependency_index", 0.0)
        ci  = data.get("cascade_influence",           0.0)
        up  = data.get("upstream_pressure",           0.0)

        structural = (
            sdi * semantic_w   +   # входящий вес → насколько узел «питается» от других
            ci  * temporal_w   +   # исходящий вес → насколько узел влияет на потомков
            investment_w * max(metrics.get("investment_score", 0.0), 0.0) -
            abs(up) * 0.1          # давление дисбаланса — штраф за асимметрию
        )

        multiplier = max(data.get("acceleration_multiplier", 1.0), 0.0)

        return (base_score + structural) * multiplier

    # ================================================================
    # BUILD NX GRAPH FROM INTERNAL STATE
    # ================================================================

    def build_nx_graph(self) -> nx.DiGraph:
        """
        Строит nx.DiGraph из self.nodes и self.edges.
        Атрибуты узлов синхронизируются из объектов Node.
        label всегда не пустой (fallback на node_id).
        """
        G = nx.DiGraph()

        for node_id, node in self.nodes.items():
            label = (node.title or node.description or node_id)[:60]
            G.add_node(
                node_id,
                label=label,
                node_type=node.node_type,
                readiness_score=node.readiness_score,
                convergence_potential=node.convergence_potential,
                scientific_score=node.scientific_score,
                investment_score=node.investment_score,
                social_score=node.social_score,
                maturity_score=node.maturity_score,
                dual_use_risk=node.dual_use_risk,
                strategic_value=node.strategic_value,
                legal_risk_score=node.legal_risk_score,
                export_control_risk=node.export_control_risk,
                is_temporal_zone=node.is_temporal_zone,
                zone_multiplier=node.zone_multiplier,
                contained_nodes=node.contained_nodes,
                acceleration_multiplier=node.acceleration_multiplier,
                structural_dependency_index=node.structural_dependency_index,
                cascade_influence=node.cascade_influence,
                upstream_pressure=node.upstream_pressure,
                forecast_score=node.forecast_score,
                # Social perception
                sentiment_review_score=node.sentiment_review_score,
                sentiment_fiction_score=node.sentiment_fiction_score,
                sentiment_forum_score=node.sentiment_forum_score,
                social_perception_score=node.social_perception_score,
                # Investment
                investment_total_usd=node.investment_total_usd,
                investment_rounds=node.investment_rounds,
                investment_data_source=node.investment_data_source,
                # Forum
                forum_post_count=node.forum_post_count,
            )

        for edge in self.edges:
            if isinstance(edge, Edge):
                src, tgt = edge.source, edge.target
                if src in self.nodes and tgt in self.nodes:
                    G.add_edge(
                        src, tgt,
                        weight=edge.total_weight if edge.total_weight > 0 else edge.semantic_similarity,
                        confidence=edge.confidence,
                        relationship_type=edge.relationship_type,
                    )
            elif isinstance(edge, dict):
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                if src in self.nodes and tgt in self.nodes:
                    G.add_edge(
                        src, tgt,
                        weight=edge.get("weights", {}).get("total", 1.0),
                    )

        return G

    # ================================================================
    # FULL GRAPH FORECAST UPDATE
    # ================================================================

    def update_graph_forecast(self) -> nx.DiGraph:
        """
        Полный пересчёт прогноза для всех узлов:
            1. Социальное восприятие (sentiment-модели)
            2. Строит nx.DiGraph
            3. Считает структурные зависимости
            4. Распространяет эффекты временных зон
            5. Считает forecast_score для каждого узла
        Возвращает обновлённый граф.
        """
        logger.info("Computing full graph forecast...")

        # Шаг 1 — Social perception (если transformers доступен)
        self.score_social_perception()

        G = self.build_nx_graph()

        self.compute_structural_dependencies(G)
        self.propagate_temporal_zone_effects(G)

        for node_id in G.nodes:
            score = self.compute_forecast_score(G, node_id)
            G.nodes[node_id]["forecast_score"] = score
            if node_id in self.nodes:
                self.nodes[node_id].forecast_score = score

        logger.info(f"Forecast computed for {G.number_of_nodes()} nodes")
        return G

    def get_3d_projection(self, G: nx.DiGraph) -> dict:
        """
        Вычисляет 3D-координаты узлов через PCA эмбеддингов.
        Возвращает {node_id: (x, y, z)}.
        Используется как альтернативный layout для visualize(mode='3d').
        """
        node_ids = [nid for nid in G.nodes if nid in self.nodes]
        if len(node_ids) < 3:
            # Fallback: spring layout + z=0
            pos2d = nx.spring_layout(G, seed=42)
            return {nid: (pos2d.get(nid, (0, 0))[0], pos2d.get(nid, (0, 0))[1], 0.0)
                    for nid in G.nodes}

        embeddings = np.array([self.nodes[nid].embedding for nid in node_ids])
        pca = PCA(n_components=3)
        coords = pca.fit_transform(embeddings)

        pos3d = {node_ids[i]: tuple(coords[i]) for i in range(len(node_ids))}

        # Узлы без эмбеддинга (не в self.nodes) получают центроид
        centroid = coords.mean(axis=0)
        for nid in G.nodes:
            if nid not in pos3d:
                pos3d[nid] = tuple(centroid)

        return pos3d

    def visualize(self, output_file: str = None, mode: str = "2d"):
        """
        Визуализация графа через Plotly.
        
        Args:
            output_file: путь для сохранения HTML (None → показать в браузере)
            mode: '2d' — spring layout (по умолчанию)
                  '3d' — PCA-проекция эмбеддингов в 3D пространство
        
        Цвет узлов: forecast_score (зелёный = высокий, красный = низкий).
        Размер узлов: readiness_score.
        Hover: label, forecast, readiness, node_type.
        """
        G = self.update_graph_forecast()

        if mode == "3d":
            self._visualize_3d(G, output_file)
        else:
            self._visualize_2d(G, output_file)

    def _visualize_2d(self, G: nx.DiGraph, output_file: str = None):
        """2D spring-layout визуализация."""
        pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos.get(u, (0, 0))
            x1, y1 = pos.get(v, (0, 0))
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))

        node_x, node_y, hover_texts, labels, sizes, colors = [], [], [], [], [], []
        for nid in G.nodes():
            x, y = pos.get(nid, (0, 0))
            data = G.nodes[nid]
            label = data.get('label') or nid[:30]
            node_x.append(x)
            node_y.append(y)
            labels.append(label[:30])
            sizes.append(10 + data.get('readiness_score', 0) * 2)
            colors.append(data.get('forecast_score', 0))
            hover_texts.append(
                f"<b>{label}</b><br>"
                f"type: {data.get('node_type', '?')}<br>"
                f"forecast: {data.get('forecast_score', 0):.3f}<br>"
                f"readiness: {data.get('readiness_score', 0):.3f}<br>"
                f"cascade_influence: {data.get('cascade_influence', 0):.2f}<br>"
                f"social_perception: {data.get('social_perception_score', 0):.2f}<br>"
                f"investment_usd: {data.get('investment_total_usd', 0):,.0f}<br>"
                f"forum_posts: {data.get('forum_post_count', 0)}"
            )

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Forecast Score"),
                line_width=2
            ),
            text=labels,
            hovertext=hover_texts,
            textposition="top center"))

        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            title="Graph Forecaster V6 — Forecast Scores (2D)",
            margin=dict(b=20, l=5, r=5, t=40),
        )

        if output_file:
            fig.write_html(output_file)
            logger.info(f"2D graph saved to {output_file}")
        else:
            fig.show()

    def _visualize_3d(self, G: nx.DiGraph, output_file: str = None):
        """3D PCA-проекция эмбеддингов."""
        pos3d = self.get_3d_projection(G)

        edge_x, edge_y, edge_z = [], [], []
        for u, v in G.edges():
            x0, y0, z0 = pos3d.get(u, (0, 0, 0))
            x1, y1, z1 = pos3d.get(v, (0, 0, 0))
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='#888', width=1),
            hoverinfo='none'))

        node_x, node_y, node_z, hover_texts, labels, sizes, colors = [], [], [], [], [], [], []
        for nid in G.nodes():
            x, y, z = pos3d.get(nid, (0, 0, 0))
            data = G.nodes[nid]
            label = data.get('label') or nid[:30]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            labels.append(label[:30])
            sizes.append(5 + data.get('readiness_score', 0) * 2)
            colors.append(data.get('forecast_score', 0))
            hover_texts.append(
                f"<b>{label}</b><br>"
                f"type: {data.get('node_type', '?')}<br>"
                f"forecast: {data.get('forecast_score', 0):.3f}<br>"
                f"readiness: {data.get('readiness_score', 0):.3f}<br>"
                f"social_perception: {data.get('social_perception_score', 0):.2f}<br>"
                f"forum_posts: {data.get('forum_post_count', 0)}"
            )

        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Forecast Score"),
                opacity=0.85,
            ),
            text=labels,
            hovertext=hover_texts,
            hoverinfo='text',
            textposition='top center'))

        fig.update_layout(
            showlegend=False,
            title="Graph Forecaster V6 — Forecast Scores (3D PCA)",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
            margin=dict(b=0, l=0, r=0, t=40),
        )

        if output_file:
            out_3d = output_file.replace(".html", "_3d.html")
            fig.write_html(out_3d)
            logger.info(f"3D graph saved to {out_3d}")
        else:
            fig.show()

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    
    config = Config()
    config.MAX_RECURSIVE_DEPTH = 3
    config.MAX_NODES_PER_LEVEL = 30
    
    gf = GraphForecasterV6(config)
    
    target = "5D optical data storage using femtosecond laser voxel writing in fused silica"
    
    logger.info("\n" + "="*80)
    logger.info("GRAPH FORECASTER V6 - CORRECTED CROSS-DOMAIN DISCOVERY")
    logger.info("="*80 + "\n")
    
    # Stage 0
    problem_tree = gf.define_target(target)
    
    # Stage 1 - Multi-source ingestion with domain extraction
    gf.ingest_all_sources(depth=config.MAX_RECURSIVE_DEPTH)
    
    # Stage 2 - Now discovers analogies based on ACTUAL discovered domains
    gf.discover_cross_domain_analogies(max_analogies_per_domain=10)
    
    logger.info(f"\nFinal statistics:")
    logger.info(f"Total nodes: {len(gf.nodes)}")
    logger.info(f"Discovered domains: {len(gf.discovered_domains)}")
    logger.info(f"Sample domains: {list(gf.discovered_domains)[:10]}")

    # Stage 4 — Full forecast update:
    #   social perception (sentiment) → структурные зависимости → временные зоны → scores
    G = gf.update_graph_forecast()

    # Top nodes by forecast score
    top_nodes = sorted(
        [(nid, G.nodes[nid].get("forecast_score", 0)) for nid in G.nodes],
        key=lambda x: x[1], reverse=True
    )[:10]
    logger.info("\nTop-10 nodes by forecast_score:")
    for nid, score in top_nodes:
        d = G.nodes[nid]
        logger.info(
            f"  {score:.3f}  {d.get('label', nid[:40])[:50]}"
            f"  | social={d.get('social_perception_score', 0):.2f}"
            f"  | invest=${d.get('investment_total_usd', 0):,.0f}"
            f"  | forums={d.get('forum_post_count', 0)}"
        )

    gf.visualize("graph_output.html", mode="2d")
    gf.visualize("graph_output.html", mode="3d")  # сохранит graph_output_3d.html