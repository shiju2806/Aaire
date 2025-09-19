"""
FormattingManager - Extracted formatting methods from RAG pipeline
Handles all text formatting, cleanup, and professional presentation tasks
"""

import re
import json
import asyncio
from typing import List, Dict, Any, Optional
import structlog
from llama_index.llms.openai import OpenAI
from openai import AsyncOpenAI

logger = structlog.get_logger()


class FormattingManager:
    """
    Manages all formatting operations for RAG pipeline responses.

    This class contains all formatting-related methods extracted from the main
    RAG pipeline to provide clean, professional text formatting and presentation.
    """

    def __init__(self, llm_client: Optional[Any] = None, llm_model: str = "gpt-4o-mini"):
        """
        Initialize the FormattingManager.

        Args:
            llm_client: OpenAI client instance for LLM operations
            llm_model: Model name to use for LLM-based formatting operations
        """
        self.llm_client = llm_client
        self.llm_model = llm_model

        # Create a dedicated formatting LLM instance
        self.formatting_llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=4000
        )

    def format_response(self, raw_content: str) -> str:
        """Pass 2: Clean formatting of the generated content with error handling"""
        try:
            logger.info("🔧 Pass 2: Starting formatting cleanup")
            if raw_content is None:
                logger.warning("🔍 Pass 2: Input content is None, returning empty string")
                return ""
            logger.info(f"🔍 Pass 2: Input content length: {len(raw_content)} characters")
            logger.info(f"🔍 Pass 2: Content preview: {raw_content[:200]}...")

            logger.info("🔧 Pass 2: Using GPT-4o-mini for enhanced formatting capabilities")

            format_prompt = f"""You are an EXPERT document formatter. Your SOLE PURPOSE is to fix ALL formatting issues while preserving EVERY word, number, and formula exactly.

===== RAW TEXT TO FORMAT =====
{raw_content}
===== END RAW TEXT =====

YOUR MISSION: Transform this into perfectly formatted text by fixing ALL these issues:

🔴 CRITICAL ISSUE #1: ORPHANED ASTERISKS
Search for these EXACT patterns and DELETE them:
- Any line containing ONLY: **
- Any line starting with: **<newline>
- Any occurrence of: **<newline><newline>**
- Standalone ** not part of bold text

🔴 CRITICAL ISSUE #2: BROKEN LIST FORMATTING
Transform EVERY instance of these patterns:

PATTERN: "-<newline>The text"
TRANSFORM TO: "• The text"

PATTERN: "- **XXX**=Description" (all on one line)
TRANSFORM TO: "• **XXX** = Description" (with spaces)

PATTERN: Multiple codes on one line like "-**061**=text-**062**=text-**063**=text"
TRANSFORM TO: Each on its own line:
• **061** = text
• **062** = text
• **063** = text

🔴 CRITICAL ISSUE #3: BROKEN HEADERS
Find and fix:
- Headers followed by orphaned ** symbols
- Headers with trailing ** at the end
- Double asterisks that aren't creating bold text

🔴 CRITICAL ISSUE #4: TRAILING ARTIFACTS
Remove ALL instances of:
- ",**" at end of sentences → remove the ,**
- ")**" at end of parentheses → keep just )
- Random ** at line ends → delete them
- Unnecessary commas before line breaks

🔴 CRITICAL ISSUE #5: SPACING AND STRUCTURE
Ensure:
- Blank line before and after each header
- Each list item on its own line
- Consistent bullet symbol (•) throughout
- Proper spacing around equals signs in formulas

STEP-BY-STEP PROCESSING ORDER:
1. Read through the ENTIRE text first
2. Identify ALL instances of the patterns above
3. Fix orphaned asterisks FIRST
4. Fix list formatting SECOND
5. Fix headers THIRD
6. Remove trailing artifacts FOURTH
7. Fix spacing LAST

VALIDATION REQUIREMENTS:
Before returning the text, verify:
✓ Zero lines contain only **
✓ Zero instances of ** at start of lines (unless bold text)
✓ All bullets use • symbol
✓ Every list item is on a separate line
✓ No ",**" or ")**" patterns remain
✓ Headers are clean without trailing symbols
✓ Proper spacing between sections

ABSOLUTE REQUIREMENTS:
1. PRESERVE every word, number, formula, and technical term
2. NEVER add or remove actual content
3. ONLY fix formatting issues
4. Use • for ALL bullet points (not -, *, or •)
5. Ensure professional, clean output

EXAMPLE TRANSFORMATIONS YOU MUST APPLY:

INPUT: "**2. Additional Considerations**\n\n**\n\nSecondary Guarantees:"
OUTPUT: "**2. Additional Considerations**\n\nSecondary Guarantees:"

INPUT: "- \nThe adjusted gross premium"
OUTPUT: "• The adjusted gross premium"

INPUT: "costs),**"
OUTPUT: "costs)"

INPUT: "-**061**=Single premium-**062**=Universal life"
OUTPUT: "• **061** = Single premium\n• **062** = Universal life"

NOW FORMAT THE TEXT - Return ONLY the perfectly formatted version:"""

            logger.info(f"🔍 Pass 2: Format prompt length: {len(format_prompt)} characters")
            logger.info("🚀 Pass 2: About to call formatting_llm.complete() with GPT-4o-mini")

            # Add timeout and more specific error handling
            import time
            start_time = time.time()

            formatted_response = self.formatting_llm.complete(format_prompt)

            end_time = time.time()
            logger.info(f"🚀 Pass 2: LLM call completed in {end_time - start_time:.2f} seconds")
            logger.info(f"🔍 Pass 2: Response object type: {type(formatted_response)}")
            logger.info(f"🔍 Pass 2: Response has text attribute: {hasattr(formatted_response, 'text')}")

            if hasattr(formatted_response, 'text'):
                logger.info(f"🔍 Pass 2: Response text length: {len(formatted_response.text)} characters")
                logger.info(f"🔍 Pass 2: Response preview: {formatted_response.text[:200]}...")

                cleaned_text = formatted_response.text.strip()

                # Optional: Second pass for stubborn issues
                if "**\n\n**" in cleaned_text or ",**" in cleaned_text or "-**" in cleaned_text:
                    logger.info("🔄 Pass 2: Detected remaining issues, attempting focused cleanup")

                    focused_prompt = f"""The formatting still has issues. Focus ONLY on these specific problems:

TEXT WITH REMAINING ISSUES:
{cleaned_text}

FIND AND FIX ONLY THESE:
1. Any "**\\n\\n**" → Delete the orphaned **
2. Any ",**" → Remove ,**
3. Any "-**XXX**=" → Change to "• **XXX** = "
4. Any line with ONLY ** → Delete entire line

CRITICAL: Keep ALL other content EXACTLY the same.

Return the corrected text:"""

                    second_response = self.formatting_llm.complete(focused_prompt)
                    if hasattr(second_response, 'text'):
                        cleaned_text = second_response.text.strip()
                        logger.info("✅ Pass 2: Focused cleanup completed")

                logger.info(f"🔍 Pass 2: Final cleaned text length: {len(cleaned_text)} characters")
                logger.info("✅ Pass 2: Formatting cleanup completed successfully")
                return cleaned_text
            else:
                logger.error("❌ Pass 2: Response object has no 'text' attribute")
                logger.info("🔄 Pass 2: Falling back to original content")
                return raw_content

        except Exception as e:
            logger.error(f"❌ Pass 2: Formatting cleanup failed with exception: {str(e)}")
            logger.error(f"❌ Pass 2: Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Pass 2: Full traceback: {traceback.format_exc()}")
            logger.info("🔄 Pass 2: Falling back to original content")
            return raw_content

    def fix_reserve_terminology(self, response: str) -> str:
        """Fix common terminology errors in reserve calculations"""
        try:
            # Fix Deferred Reserve -> Deterministic Reserve
            response = re.sub(r'\bDeferred Reserve\b', 'Deterministic Reserve', response, flags=re.IGNORECASE)
            response = re.sub(r'\bDeferred Reserves\b', 'Deterministic Reserves', response, flags=re.IGNORECASE)

            # Ensure DR is correctly defined
            response = re.sub(r'\bDR\s*=\s*Deferred', 'DR = Deterministic', response, flags=re.IGNORECASE)

            # Fix Scenario Reserve -> Stochastic Reserve (if needed)
            # Note: Scenario Reserve is sometimes acceptable, but Stochastic is more precise
            response = re.sub(r'\bScenario Reserve\s*\(SR\)', 'Stochastic Reserve (SR)', response)

            # Fix inconsistent header formatting
            response = re.sub(r'\*\*([A-Z][a-z]+.*?):\*\*\s*([A-Z])', r'\n**\1**\n\2', response)
            response = re.sub(r'•\s*\*\*(.*?):\*\*', r'• **\1:**', response)

            logger.info("Fixed reserve terminology and formatting")
            return response

        except Exception as e:
            logger.warning(f"Failed to fix terminology: {str(e)}")
            return response

    def preserve_formulas(self, response: str, documents: List[Dict]) -> str:
        """Extract and preserve mathematical formulas from documents"""
        try:
            # Extract all mathematical content from documents
            all_content = " ".join([doc['content'] for doc in documents])

            formula_prompt = f"""Extract ALL mathematical formulas, equations, and calculations from this content.

Content:
{all_content[:4000]}

Find every formula, equation, calculation method, or mathematical expression.
Preserve the exact notation including LaTeX markup, variables, subscripts, etc.

Output each formula with a brief description:
1. [Description]: [Exact formula as written]
2. [Description]: [Exact formula as written]

If no formulas found, respond with "NO_FORMULAS".

Extracted formulas:"""

            formula_response = self.formatting_llm.complete(formula_prompt)
            extracted_formulas = formula_response.text.strip()

            logger.info(f"🧮 FORMULA EXTRACTION RESULT: {extracted_formulas[:300]}...")

            if extracted_formulas and "NO_FORMULAS" not in extracted_formulas:
                # Check if formulas are already well-represented in response
                formula_check_prompt = f"""Are these mathematical formulas adequately represented in the response?

Response:
{response[:1000]}...

Formulas from documents:
{extracted_formulas}

If key formulas are missing or oversimplified, respond with "ADD_FORMULAS".
If formulas are well-represented, respond with "FORMULAS_ADEQUATE".

Assessment:"""

                formula_check = self.formatting_llm.complete(formula_check_prompt)

                logger.info(f"🧮 FORMULA CHECK RESULT: {formula_check.text.strip()}")

                if "ADD_FORMULAS" in formula_check.text:
                    # Add mathematical formulas section
                    enhanced_response = f"""{response}

## Mathematical Formulas and Calculations

{extracted_formulas}"""

                    logger.info("✅ Added mathematical formulas section")
                    return enhanced_response
                else:
                    logger.info("❌ Formulas determined to be adequately represented")
            else:
                logger.info("❌ No formulas extracted or NO_FORMULAS returned")

            return response

        except Exception as e:
            logger.warning(f"Formula preservation failed: {str(e)}")
            return response

    def apply_llm_formatting_fix(self, response: str, detected_issues: list) -> str:
        """Apply LLM-based formatting correction for detected issues"""

        issues_description = ', '.join(detected_issues)

        correction_prompt = f"""Fix the formatting issues in this insurance/actuarial response.

DETECTED ISSUES: {issues_description}

SPECIFIC FIXES NEEDED:
1. Put blank line BEFORE each **numbered item** (like **1.** or **2.**)
2. Separate numbered list items (1. 2. 3.) onto different lines
3. Add line break after "Where:" before definitions
4. Fix bold formatting artifacts like **includes or **excludes
5. Keep ALL formulas and mathematical content intact

EXAMPLE OF CORRECT FORMAT:
**Section Title**

Regular paragraph text here.

**1.** First numbered point

**2.** Second numbered point

Formula: NPR = APV(Benefits) - APV(Premiums)

Where:
- NPR = Net Premium Reserve
- APV = Actuarial Present Value

Original text to fix:
{response}

Provide the corrected version:"""

        try:
            corrected = self.formatting_llm.complete(correction_prompt, temperature=0.1)
            logger.info("🔧 Applied LLM-based formatting correction")
            return corrected.text.strip()
        except Exception as e:
            logger.warning(f"LLM formatting correction failed: {e}")
            return response

    def generate_structured_response(self, query: str, context: str) -> str:
        """Generate response with structured JSON output for consistent formatting"""

        structured_prompt = f"""You are AAIRE, an insurance accounting expert.

Question: {query}

Context: {context}

Generate a response in this EXACT JSON structure (be precise with formulas):
{{
    "summary": "Brief 2-3 sentence overview",
    "sections": [
        {{
            "title": "Section heading",
            "content": "Detailed explanation",
            "formulas": [
                {{
                    "name": "Reserve Calculation",
                    "latex": "R_t = PV(benefits_t) - PV(premiums_t)",
                    "readable": "R(t) = PV(benefits at time t) - PV(premiums at time t)",
                    "components": {{
                        "R_t": "Reserve at time t",
                        "PV": "Present Value function"
                    }}
                }}
            ],
            "numbered_items": ["Step 1 description", "Step 2 description"]
        }}
    ],
    "key_values": {{
        "rates": ["90% confidence level", "2.5% discount rate"],
        "references": ["ASC 944-40-25-25", "IFRS 17.32"]
    }}
}}

Ensure ALL mathematical notation is included in both latex and readable formats.
Only return the JSON - no other text."""

        try:
            response = self.formatting_llm.complete(structured_prompt, temperature=0.2)
            structured_data = json.loads(response.text.strip())
            logger.info("✅ Successfully generated structured JSON response")
            return self.structured_to_markdown(structured_data)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            # Fallback to existing correction method
            return self.apply_llm_formatting_fix(response.text, ["JSON structure invalid"])
        except Exception as e:
            logger.warning(f"Structured generation failed: {e}")
            raise

    def structured_to_markdown(self, data: Dict) -> str:
        """Convert structured JSON to formatted markdown with proper formula handling"""

        output = []

        # Summary
        if 'summary' in data:
            output.append(f"{data['summary']}\n")

        # Process each section
        for section in data.get('sections', []):
            # Section title
            output.append(f"\n**{section['title']}**\n")

            # Content
            if 'content' in section:
                output.append(f"{section['content']}\n")

            # Formulas - with special handling
            if 'formulas' in section:
                output.append("\n**Key Formulas:**\n")
                for formula in section['formulas']:
                    # Use readable format as primary
                    output.append(f"\n• {formula['name']}:\n")
                    output.append(f"  {formula['readable']}\n")

                    # Add component definitions if present
                    if 'components' in formula:
                        output.append("  Where:\n")
                        for var, desc in formula['components'].items():
                            output.append(f"  - {var} = {desc}\n")

            # Numbered items with proper spacing
            if 'numbered_items' in section:
                output.append("\n")
                for i, item in enumerate(section['numbered_items'], 1):
                    output.append(f"**{i}.** {item}\n\n")

        # Key values
        if 'key_values' in data:
            output.append("\n**Important Values:**\n")
            for category, values in data['key_values'].items():
                for value in values:
                    output.append(f"• {value}\n")

        return ''.join(output)

    def validate_formula_formatting(self, response: str) -> bool:
        """Check if formulas are properly formatted"""

        validation_prompt = f"""Check if this text has properly formatted formulas:

{response[:1000]}

Look for:
1. LaTeX notation that wasn't converted (\\sum, \\times, _{{subscript}})
2. Unreadable mathematical expressions
3. Complex subscripts not converted to parentheses

Reply with just: VALID or NEEDS_FIXING"""

        try:
            result = self.formatting_llm.complete(validation_prompt, temperature=0)
            return "VALID" in result.text.upper()
        except Exception as e:
            logger.warning(f"Formula validation failed: {e}")
            return True  # Default to assuming it's valid

    def normalize_spacing(self, response: str) -> str:
        """Simplified cleanup to fix persistent formatting issues without breaking content"""

        # Step 1: Fix the most critical issues - text running together
        # Ensure proper spacing around numbered sections (1., 2., etc.)
        result = re.sub(r'([a-z\.])(\d+\.)', r'\1\n\n\2', response)

        # Fix text immediately following bold markers without space
        result = re.sub(r'\*\*([^*]+)\*\*([A-Z])', r'**\1**\n\n\2', result)

        # Step 2: Conservative bold formatting cleanup (only clear issues)
        # Fix excessive asterisks
        result = re.sub(r'\*{3,}', '**', result)

        # Only convert numbered sections to headers if they're clearly headers
        result = re.sub(r'\*\*(\d+\.\s*[A-Z][^*]{10,}?)\*\*', r'## \1', result)

        # Step 3: Clean up whitespace issues
        # Remove extra spaces but preserve line breaks
        result = re.sub(r'[ \t]+', ' ', result)

        # Control excessive newlines
        result = re.sub(r'\n{4,}', '\n\n\n', result)

        # Step 4: Fix spacing after punctuation
        result = re.sub(r'([\.!?])([A-Z])', r'\1 \2', result)

        # Step 5: Ensure proper spacing around list items
        result = re.sub(r'([^\n])\n-\s', r'\1\n\n- ', result)

        result = result.strip()
        return result

    async def apply_professional_formatting(self, response: str, query: str) -> str:
        """Apply ChatGPT-style professional formatting to responses"""
        try:
            logger.info("📝 Applying professional formatting to response")

            # Use LLM to reformat the content professionally
            system_prompt = """You are a formatting expert. Rewrite the provided technical content with:

1. Clear visual hierarchy using emojis as section markers (🔹 for main points, ✅ for summary, 📌 for key points)
2. Numbered sections with proper spacing
3. Sub-points using (a), (b), (c) or bullet points
4. Bold only for key terms and headers (use sparingly)
5. Clean spacing between sections
6. Professional, conversational tone
7. Examples where helpful
8. A clear summary at the end

Format like high-quality ChatGPT responses - clean, organized, and easy to scan.
Keep all technical accuracy but improve readability dramatically.
Do NOT add unnecessary information - only reformat what's provided."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nContent to reformat:\n{response}"}
            ]

            formatted_response = await self.llm_client.achat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=4000
            )

            result = formatted_response.choices[0].message.content

            # Apply final clean-up
            result = self.final_formatting_cleanup(result)

            logger.info("✅ Professional formatting applied successfully")
            return result

        except Exception as e:
            logger.warning(f"Could not apply professional formatting: {e}")
            # Fall back to basic cleanup
            return self.basic_professional_format(response)

    def final_formatting_cleanup(self, text: str) -> str:
        """Final cleanup for professional formatting"""

        # Ensure consistent spacing
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # Ensure space after emoji markers
        text = re.sub(r'(🔹|✅|📌|📍|🎯|💡|⚠️|❌|✔️)([A-Za-z])', r'\1 \2', text)

        # Clean up any remaining formatting issues
        text = re.sub(r'\*{3,}', '**', text)

        return text.strip()

    def apply_simple_professional_formatting(self, response: str) -> str:
        """Simple, reliable professional formatting without API calls"""

        # Split into paragraphs
        paragraphs = response.split('\n\n')
        formatted_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Detect and format different types of content
            if re.match(r'^\d+\.', para):
                # Main numbered section - add emoji marker
                para = re.sub(r'^(\d+\.)', r'🔹 \1', para)
                # Ensure space after the number
                para = re.sub(r'^(🔹\s*\d+\.)([A-Z])', r'\1 \2', para)
            elif re.match(r'^##?\s', para):
                # Already a header, leave it
                pass
            elif 'summary' in para.lower() or 'conclusion' in para.lower():
                # Summary sections
                if not para.startswith('✅'):
                    para = f"✅ {para}"
            elif re.match(r'^-\s', para):
                # Bullet point
                para = re.sub(r'^-\s', '• ', para)

            # Fix excessive bold - only keep for important terms
            if para.count('**') > 6:
                # Too much bold, reduce it
                para = re.sub(r'\*\*([^*]{1,15})\*\*', r'\1', para)

            formatted_paragraphs.append(para)

        result = '\n\n'.join(formatted_paragraphs)

        # Final cleanup
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r'\*{3,}', '**', result)

        return result.strip()

    def basic_professional_format(self, response: str) -> str:
        """Fallback basic professional formatting without LLM"""

        lines = response.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue

            # Add emoji markers for numbered items
            if re.match(r'^\d+\.', line):
                # Main numbered sections
                line = f"🔹 {line}"
            elif re.match(r'^[a-z]\)', line):
                # Sub-points
                line = f"  • {line[2:].strip()}"
            elif line.startswith('-'):
                # Bullet points
                line = f"  • {line[1:].strip()}"

            formatted_lines.append(line)

        result = '\n'.join(formatted_lines)

        # Add proper spacing
        result = re.sub(r'(🔹[^\n]+)\n([^🔹\n])', r'\1\n\n\2', result)

        # Add summary marker if there's a summary section
        result = re.sub(r'(Summary|Conclusion|Final)', r'✅ \1', result, flags=re.IGNORECASE)

        return result.strip()


def create_formatting_manager(llm_client: Optional[Any] = None, llm_model: str = "gpt-4o-mini") -> FormattingManager:
    """
    Factory function to create a FormattingManager instance.

    Args:
        llm_client: OpenAI client instance for LLM operations
        llm_model: Model name to use for LLM-based formatting operations

    Returns:
        FormattingManager: Configured formatting manager instance
    """
    return FormattingManager(llm_client=llm_client, llm_model=llm_model)