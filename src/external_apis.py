"""
External API Integrations for AAIRE - MVP Weeks 5-6
SEC EDGAR and FRED API connectors for free data sources
"""

import os
import asyncio
import aiohttp
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog
import yaml

from llama_index.core import Document

logger = structlog.get_logger()

@dataclass
class APIResponse:
    success: bool
    data: Any
    error: Optional[str] = None
    source: str = ""

class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_called = 0.0
    
    async def acquire(self):
        elapsed = asyncio.get_event_loop().time() - self.last_called
        left_to_wait = self.min_interval - elapsed
        if left_to_wait > 0:
            await asyncio.sleep(left_to_wait)
        self.last_called = asyncio.get_event_loop().time()

class SECEdgarConnector:
    """SEC EDGAR API connector for real accounting policy examples"""
    
    def __init__(self):
        # Load configuration
        with open('config/data_sources.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['free_apis']['sec_edgar']
        self.base_url = self.config['base_url']
        self.user_agent = self.config['user_agent']
        
        # Rate limiter: 10 requests per second
        self.rate_limiter = RateLimiter(10.0)
        
        # Session for connection pooling
        self.session = None
        
        logger.info("SEC EDGAR connector initialized", base_url=self.base_url)
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def search_companies(self, query: str, limit: int = 10) -> APIResponse:
        """Search for companies in EDGAR database"""
        try:
            await self.rate_limiter.acquire()
            session = await self._get_session()
            
            # EDGAR company search endpoint
            url = f"{self.base_url}/company_tickers.json"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter companies based on query
                    filtered_companies = []
                    for key, company in data.items():
                        if query.lower() in company.get('title', '').lower():
                            filtered_companies.append(company)
                            if len(filtered_companies) >= limit:
                                break
                    
                    return APIResponse(
                        success=True,
                        data=filtered_companies,
                        source="SEC EDGAR"
                    )
                else:
                    return APIResponse(
                        success=False,
                        error=f"HTTP {response.status}",
                        source="SEC EDGAR"
                    )
                    
        except Exception as e:
            logger.error("SEC EDGAR search failed", exception_details=str(e))
            return APIResponse(
                success=False,
                error=str(e),
                source="SEC EDGAR"
            )
    
    async def get_company_filings(self, cik: str, form_type: str = "10-K", limit: int = 5) -> APIResponse:
        """Get recent filings for a company"""
        try:
            await self.rate_limiter.acquire()
            session = await self._get_session()
            
            # Format CIK with leading zeros
            cik = cik.zfill(10)
            url = f"{self.base_url}/submissions/CIK{cik}.json"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract filings of the specified type
                    filings = data.get('filings', {}).get('recent', {})
                    
                    filtered_filings = []
                    for i, form in enumerate(filings.get('form', [])):
                        if form == form_type and len(filtered_filings) < limit:
                            filing = {
                                'form': form,
                                'filingDate': filings.get('filingDate', [])[i],
                                'accessionNumber': filings.get('accessionNumber', [])[i],
                                'primaryDocument': filings.get('primaryDocument', [])[i],
                                'description': filings.get('primaryDocDescription', [])[i]
                            }
                            filtered_filings.append(filing)
                    
                    return APIResponse(
                        success=True,
                        data=filtered_filings,
                        source="SEC EDGAR"
                    )
                else:
                    return APIResponse(
                        success=False,
                        error=f"HTTP {response.status}",
                        source="SEC EDGAR"
                    )
                    
        except Exception as e:
            logger.error("SEC EDGAR filings request failed", exception_details=str(e))
            return APIResponse(
                success=False,
                error=str(e),
                source="SEC EDGAR"
            )
    
    async def get_filing_content(self, accession_number: str, document: str) -> APIResponse:
        """Get content of a specific filing"""
        try:
            await self.rate_limiter.acquire()
            session = await self._get_session()
            
            # Build filing URL
            accession_clean = accession_number.replace('-', '')
            url = f"https://www.sec.gov/Archives/edgar/data/{accession_clean}/{accession_number}/{document}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return APIResponse(
                        success=True,
                        data=content,
                        source="SEC EDGAR"
                    )
                else:
                    return APIResponse(
                        success=False,
                        error=f"HTTP {response.status}",
                        source="SEC EDGAR"
                    )
                    
        except Exception as e:
            logger.error("SEC EDGAR content request failed", exception_details=str(e))
            return APIResponse(
                success=False,
                error=str(e),
                source="SEC EDGAR"
            )
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class FREDAPIConnector:
    """Federal Reserve Economic Data (FRED) API connector"""
    
    def __init__(self):
        # Load configuration
        with open('config/data_sources.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['free_apis']['fred_api']
        self.base_url = self.config['base_url']
        self.api_key = os.getenv('FRED_API_KEY')
        
        if not self.api_key:
            logger.warning("FRED API key not found in environment variables")
        
        # Rate limiter: 120 requests per minute = 2 per second
        self.rate_limiter = RateLimiter(2.0)
        
        self.session = None
        
        logger.info("FRED API connector initialized", base_url=self.base_url)
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_series_data(self, series_id: str, limit: int = 100) -> APIResponse:
        """Get data for a specific FRED series"""
        if not self.api_key:
            return APIResponse(
                success=False,
                error="FRED API key not configured",
                source="FRED"
            )
        
        try:
            await self.rate_limiter.acquire()
            session = await self._get_session()
            
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'  # Most recent first
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return APIResponse(
                        success=True,
                        data=data.get('observations', []),
                        source="FRED"
                    )
                else:
                    return APIResponse(
                        success=False,
                        error=f"HTTP {response.status}",
                        source="FRED"
                    )
                    
        except Exception as e:
            logger.error("FRED API request failed", exception_details=str(e))
            return APIResponse(
                success=False,
                error=str(e),
                source="FRED"
            )
    
    async def search_series(self, search_text: str, limit: int = 25) -> APIResponse:
        """Search for FRED data series"""
        if not self.api_key:
            return APIResponse(
                success=False,
                error="FRED API key not configured",
                source="FRED"
            )
        
        try:
            await self.rate_limiter.acquire()
            session = await self._get_session()
            
            url = f"{self.base_url}/series/search"
            params = {
                'search_text': search_text,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return APIResponse(
                        success=True,
                        data=data.get('seriess', []),
                        source="FRED"
                    )
                else:
                    return APIResponse(
                        success=False,
                        error=f"HTTP {response.status}",
                        source="FRED"
                    )
                    
        except Exception as e:
            logger.error("FRED search failed", exception_details=str(e))
            return APIResponse(
                success=False,
                error=str(e),
                source="FRED"
            )
    
    async def get_interest_rates(self) -> APIResponse:
        """Get current interest rate data for insurance discounting"""
        
        # Key interest rate series for insurance/actuarial use
        rate_series = {
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            'DGS30': '30-Year Treasury Constant Maturity Rate',
            'FEDFUNDS': 'Federal Funds Rate',
            'DGS5': '5-Year Treasury Constant Maturity Rate'
        }
        
        results = {}
        
        for series_id, description in rate_series.items():
            response = await self.get_series_data(series_id, limit=1)
            if response.success and response.data:
                latest = response.data[0]
                results[series_id] = {
                    'description': description,
                    'latest_value': latest.get('value'),
                    'date': latest.get('date')
                }
        
        return APIResponse(
            success=True,
            data=results,
            source="FRED"
        )
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class ExternalAPIManager:
    """Manager for all external API integrations"""
    
    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline
        self.sec_connector = SECEdgarConnector()
        self.fred_connector = FREDAPIConnector()
        
        # Refresh jobs tracking
        self.refresh_jobs = {}
        
        logger.info("External API manager initialized")
    
    async def refresh_all(self) -> str:
        """Refresh data from all external sources"""
        job_id = str(uuid.uuid4())
        
        self.refresh_jobs[job_id] = {
            'status': 'started',
            'started_at': datetime.utcnow().isoformat(),
            'progress': 0,
            'sources': ['SEC_EDGAR', 'FRED']
        }
        
        # Start background refresh
        asyncio.create_task(self._refresh_all_async(job_id))
        
        return job_id
    
    async def _refresh_all_async(self, job_id: str):
        """Background task to refresh all external data"""
        try:
            # Refresh SEC EDGAR data
            await self._refresh_sec_data(job_id)
            self.refresh_jobs[job_id]['progress'] = 50
            
            # Refresh FRED data
            await self._refresh_fred_data(job_id)
            self.refresh_jobs[job_id]['progress'] = 100
            
            self.refresh_jobs[job_id]['status'] = 'completed'
            self.refresh_jobs[job_id]['completed_at'] = datetime.utcnow().isoformat()
            
        except Exception as e:
            self.refresh_jobs[job_id]['status'] = 'failed'
            self.refresh_jobs[job_id]['error'] = str(e)
            logger.error("External data refresh failed", job_id=job_id, exception_details=str(e))
    
    async def _refresh_sec_data(self, job_id: str):
        """Refresh SEC EDGAR data"""
        try:
            # Get major insurance companies
            insurance_companies = ['AIG', 'Berkshire Hathaway', 'Progressive', 'Allstate']
            
            documents = []
            
            for company in insurance_companies:
                # Search for company
                search_response = await self.sec_connector.search_companies(company, limit=1)
                
                if search_response.success and search_response.data:
                    company_data = search_response.data[0]
                    cik = company_data.get('cik_str')
                    
                    if cik:
                        # Get recent 10-K filings
                        filings_response = await self.sec_connector.get_company_filings(cik, '10-K', limit=1)
                        
                        if filings_response.success and filings_response.data:
                            for filing in filings_response.data:
                                # Create document for the filing metadata
                                doc_text = f"""
                                Company: {company}
                                Form: {filing['form']}
                                Filing Date: {filing['filingDate']}
                                Description: {filing.get('description', 'N/A')}
                                
                                This filing contains accounting policies and disclosures relevant to insurance accounting under US GAAP.
                                """
                                
                                document = Document(
                                    text=doc_text,
                                    metadata={
                                        'source': 'SEC_EDGAR',
                                        'company': company,
                                        'form_type': filing['form'],
                                        'filing_date': filing['filingDate'],
                                        'accession_number': filing['accessionNumber']
                                    }
                                )
                                documents.append(document)
            
            # Add documents to RAG pipeline (single index with metadata)
            if documents and self.rag_pipeline:
                await self.rag_pipeline.add_documents(documents, 'us_gaap')
                
            logger.info("SEC EDGAR data refreshed", document_count=len(documents))
            
        except Exception as e:
            logger.error("SEC data refresh failed", exception_details=str(e))
            raise
    
    async def _refresh_fred_data(self, job_id: str):
        """Refresh FRED economic data"""
        try:
            # Get current interest rates
            rates_response = await self.fred_connector.get_interest_rates()
            
            documents = []
            
            if rates_response.success:
                # Create document with current rate information
                rates_text = "Current Interest Rates for Insurance Discounting:\\n\\n"
                
                for series_id, rate_info in rates_response.data.items():
                    rates_text += f"{rate_info['description']}: {rate_info['latest_value']}% as of {rate_info['date']}\\n"
                
                rates_text += """
                
                These rates are commonly used in insurance and actuarial calculations for:
                - Discounting future cash flows
                - Reserve calculations
                - Asset-liability matching
                - IFRS 17 discount rate determinations
                """
                
                document = Document(
                    text=rates_text,
                    metadata={
                        'source': 'FRED',
                        'data_type': 'interest_rates',
                        'updated_at': datetime.utcnow().isoformat()
                    }
                )
                documents.append(document)
            
            # Add documents to RAG pipeline (single index with metadata)
            if documents and self.rag_pipeline:
                await self.rag_pipeline.add_documents(documents, 'actuarial')
            
            logger.info("FRED data refreshed", document_count=len(documents))
            
        except Exception as e:
            logger.error("FRED data refresh failed", exception_details=str(e))
            raise
    
    async def get_refresh_status(self, job_id: str) -> Dict[str, Any]:
        """Get refresh job status"""
        if job_id not in self.refresh_jobs:
            raise ValueError("Job not found")
        
        return self.refresh_jobs[job_id]
    
    async def close(self):
        """Close all connections"""
        await self.sec_connector.close()
        await self.fred_connector.close()