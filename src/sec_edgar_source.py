"""
SEC EDGAR API Integration - Clean Implementation
Fetches public company filings for AAIRE
"""

import asyncio
import aiohttp
import structlog
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = structlog.get_logger()

class SECEdgarSource:
    """Clean SEC EDGAR API client"""
    
    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": "AAIRE Insurance Assistant (contact@aaire.xyz)",
            "Accept": "application/json"
        }
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_company(self, query: str) -> List[Dict[str, Any]]:
        """Search for companies by name or ticker"""
        try:
            # Use SEC company tickers endpoint - try alternative
            url = "https://www.sec.gov/files/company_tickers.json"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"SEC API error: {response.status}")
                    return []
                
                data = await response.json()
                
                # Filter companies matching query
                companies = []
                query_lower = query.lower()
                
                for key, company in data.items():
                    company_name = company.get('title', '').lower()
                    ticker = company.get('ticker', '').lower()
                    
                    if query_lower in company_name or query_lower in ticker:
                        companies.append({
                            'cik': f"{int(company.get('cik_str', 0)):010d}",
                            'ticker': company.get('ticker', ''),
                            'title': company.get('title', ''),
                            'name': company.get('title', '')
                        })
                        
                        if len(companies) >= 10:  # Limit results
                            break
                
                return companies
                
        except Exception as e:
            logger.error(f"Company search failed: {e}")
            return []
    
    async def get_company_filings(self, cik: str, form_types: List[str] = None, years: List[int] = None) -> List[Dict[str, Any]]:
        """Get recent filings for a company"""
        try:
            # Clean CIK format
            cik_clean = cik.replace('CIK', '').zfill(10)
            
            url = f"{self.base_url}/submissions/CIK{cik_clean}.json"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"SEC filings API error: {response.status}")
                    return []
                
                data = await response.json()
                
                # Extract recent filings
                recent_filings = data.get('filings', {}).get('recent', {})
                
                filings = []
                form_list = recent_filings.get('form', [])
                filing_dates = recent_filings.get('filingDate', [])
                accession_numbers = recent_filings.get('accessionNumber', [])
                
                for i, form_type in enumerate(form_list):
                    if i >= len(filing_dates) or i >= len(accession_numbers):
                        break
                    
                    # Filter by form types if specified
                    if form_types and form_type not in form_types:
                        continue
                    
                    # Filter by years if specified
                    filing_date = filing_dates[i]
                    if years and int(filing_date[:4]) not in years:
                        continue
                    
                    filings.append({
                        'form_type': form_type,
                        'filing_date': filing_date,
                        'accession_number': accession_numbers[i],
                        'company_name': data.get('name', ''),
                        'ticker': data.get('tickers', [''])[0] if data.get('tickers') else '',
                        'cik': cik_clean
                    })
                    
                    if len(filings) >= 20:  # Limit results
                        break
                
                return filings
                
        except Exception as e:
            logger.error(f"Filing retrieval failed: {e}")
            return []
    
    async def download_filing_content(self, filing_info: Dict[str, Any]) -> Optional[str]:
        """Download filing content"""
        try:
            accession_number = filing_info['accession_number'].replace('-', '')
            cik = filing_info['cik']
            
            # Try main filing document first
            filing_url = f"{self.base_url}/Archives/edgar/data/{int(cik)}/{accession_number}/{filing_info['accession_number']}.txt"
            
            async with self.session.get(filing_url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Extract meaningful content (remove SEC headers/footers)
                    lines = content.split('\n')
                    clean_lines = []
                    in_document = False
                    
                    for line in lines:
                        if '<DOCUMENT>' in line:
                            in_document = True
                        elif '</DOCUMENT>' in line:
                            in_document = False
                        elif in_document and line.strip():
                            # Remove HTML tags for cleaner text
                            clean_line = line.strip()
                            if not clean_line.startswith('<') or 'text' in clean_line.lower():
                                clean_lines.append(clean_line)
                    
                    return '\n'.join(clean_lines[:1000])  # Limit content size
                
                return None
                
        except Exception as e:
            logger.error(f"Filing download failed: {e}")
            return None