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
                primary_documents = recent_filings.get('primaryDocument', [])

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
                        'primary_document': primary_documents[i] if i < len(primary_documents) else None,
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
            primary_doc = filing_info.get('primary_document')

            # SEC EDGAR URL format: https://www.sec.gov/Archives/edgar/data/CIK/ACCESSION/PRIMARY_DOC
            # Example: https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm

            if not primary_doc:
                logger.error(f"No primary_document in filing_info: {filing_info}")
                return None

            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_doc}"
            logger.info(f"Attempting to download: {doc_url}")

            async with self.session.get(doc_url) as response:
                logger.info(f"SEC response status: {response.status}")

                if response.status == 200:
                    content = await response.text()
                    logger.info(f"Downloaded {len(content)} characters")

                    # Extract meaningful text content
                    # For HTML files, extract text between tags
                    # For TXT files with SGML, extract from DOCUMENT sections

                    if '<DOCUMENT>' in content:
                        # SGML format - extract document sections
                        lines = content.split('\n')
                        clean_lines = []
                        in_document = False

                        for line in lines:
                            if '<DOCUMENT>' in line:
                                in_document = True
                            elif '</DOCUMENT>' in line:
                                in_document = False
                            elif in_document and line.strip():
                                # Skip SGML headers
                                if not line.startswith('<') or any(tag in line.lower() for tag in ['<text', '<html', '<p', '<div', '<span']):
                                    clean_lines.append(line.strip())

                        result = '\n'.join(clean_lines[:2000])  # Increased limit for better content
                    else:
                        # HTML format - strip tags for plain text
                        import re
                        # Remove script/style tags and their content
                        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
                        # Remove HTML tags but keep text
                        content = re.sub(r'<[^>]+>', ' ', content)
                        # Clean up whitespace
                        content = re.sub(r'\s+', ' ', content)
                        result = content[:10000].strip()  # First 10k chars

                    logger.info(f"Extracted {len(result)} characters of clean content")
                    return result if result.strip() else None
                else:
                    logger.error(f"SEC returned status {response.status} for {doc_url}")
                    return None

        except Exception as e:
            logger.error(f"Filing download failed: {e}")
            return None