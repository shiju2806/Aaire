"""
Conversational Workflow Engine for AAIRE
Guides users through step-by-step accounting processes
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import structlog

logger = structlog.get_logger()

class WorkflowStep:
    def __init__(self, step_id: str, step_data: Dict[str, Any]):
        self.id = step_id
        self.title = step_data.get('title', '')
        self.description = step_data.get('description', '')
        self.instruction = step_data.get('instruction', '')
        self.input_type = step_data.get('input_type', 'text')  # text, choice, file, calculation
        self.choices = step_data.get('choices', [])
        self.validation = step_data.get('validation', {})
        self.next_step = step_data.get('next_step')
        self.conditional_next = step_data.get('conditional_next', {})
        self.help_text = step_data.get('help_text', '')
        self.required = step_data.get('required', True)

class WorkflowTemplate:
    def __init__(self, template_data: Dict[str, Any]):
        self.id = template_data.get('id', '')
        self.name = template_data.get('name', '')
        self.description = template_data.get('description', '')
        self.category = template_data.get('category', 'general')
        self.estimated_time = template_data.get('estimated_time', '10-15 minutes')
        self.difficulty = template_data.get('difficulty', 'intermediate')
        self.steps = {
            step_id: WorkflowStep(step_id, step_data) 
            for step_id, step_data in template_data.get('steps', {}).items()
        }
        self.start_step = template_data.get('start_step', 'step1')

class WorkflowSession:
    def __init__(self, session_id: str, template: WorkflowTemplate, user_id: str = "demo-user"):
        self.session_id = session_id
        self.template = template
        self.user_id = user_id
        self.current_step = template.start_step
        self.completed_steps = []
        self.step_data = {}
        self.started_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.status = 'active'  # active, completed, abandoned

class WorkflowEngine:
    def __init__(self, templates_dir: str = "data/workflows"):
        """Initialize workflow engine"""
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates = {}
        self.active_sessions = {}
        
        # Load workflow templates
        asyncio.create_task(self._load_templates())
        
        logger.info("Workflow engine initialized", templates_dir=str(self.templates_dir))
    
    async def _load_templates(self):
        """Load workflow templates from files"""
        try:
            # Create default templates if they don't exist
            await self._create_default_templates()
            
            # Load all template files
            for template_file in self.templates_dir.glob("*.json"):
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                        template = WorkflowTemplate(template_data)
                        self.templates[template.id] = template
                        logger.info("Loaded workflow template", template_id=template.id, name=template.name)
                except Exception as e:
                    logger.error("Failed to load template", file=str(template_file), error=str(e))
        
        except Exception as e:
            logger.error("Failed to load workflow templates", error=str(e))
    
    async def _create_default_templates(self):
        """Create default accounting workflow templates"""
        
        # Revenue Recognition Workflow
        revenue_workflow = {
            "id": "revenue_recognition_asc606",
            "name": "Revenue Recognition Analysis (ASC 606)",
            "description": "Step-by-step guide to analyze revenue recognition under ASC 606",
            "category": "revenue",
            "estimated_time": "15-20 minutes",
            "difficulty": "intermediate",
            "start_step": "step1",
            "steps": {
                "step1": {
                    "title": "Contract Identification",
                    "description": "Identify the contract with the customer",
                    "instruction": "Do you have a written or verbal agreement with enforceable rights and obligations?",
                    "input_type": "choice",
                    "choices": [
                        {"value": "written", "label": "Written contract"},
                        {"value": "verbal", "label": "Verbal agreement"},
                        {"value": "none", "label": "No formal agreement"}
                    ],
                    "conditional_next": {
                        "written": "step2",
                        "verbal": "step2",
                        "none": "step_no_contract"
                    },
                    "help_text": "A contract exists when there are enforceable rights and obligations between parties."
                },
                "step2": {
                    "title": "Performance Obligations",
                    "description": "Identify distinct performance obligations",
                    "instruction": "List the distinct goods or services promised in the contract (one per line):",
                    "input_type": "text",
                    "next_step": "step3",
                    "help_text": "A performance obligation is distinct if the customer can benefit from it and it's separately identifiable."
                },
                "step3": {
                    "title": "Transaction Price",
                    "description": "Determine the transaction price",
                    "instruction": "What is the total transaction price (including variable consideration)?",
                    "input_type": "text",
                    "validation": {"type": "number", "min": 0},
                    "next_step": "step4",
                    "help_text": "Include estimates of variable consideration if you expect to be entitled to it."
                },
                "step4": {
                    "title": "Price Allocation",
                    "description": "Allocate transaction price to performance obligations",
                    "instruction": "How will you allocate the transaction price?",
                    "input_type": "choice",
                    "choices": [
                        {"value": "standalone", "label": "Based on standalone selling prices"},
                        {"value": "estimate", "label": "Estimate when standalone prices unavailable"},
                        {"value": "residual", "label": "Residual approach"}
                    ],
                    "next_step": "step5"
                },
                "step5": {
                    "title": "Revenue Recognition Timing",
                    "description": "Determine when to recognize revenue",
                    "instruction": "When is control transferred to the customer?",
                    "input_type": "choice",
                    "choices": [
                        {"value": "point_in_time", "label": "At a point in time"},
                        {"value": "over_time", "label": "Over time"}
                    ],
                    "conditional_next": {
                        "point_in_time": "step_point_in_time",
                        "over_time": "step_over_time"
                    }
                },
                "step_point_in_time": {
                    "title": "Point-in-Time Recognition",
                    "description": "Identify the specific point when control transfers",
                    "instruction": "Describe when control transfers (e.g., delivery, customer acceptance):",
                    "input_type": "text",
                    "next_step": "step_summary"
                },
                "step_over_time": {
                    "title": "Over-Time Recognition",
                    "description": "Determine progress measurement method",
                    "instruction": "How will you measure progress?",
                    "input_type": "choice",
                    "choices": [
                        {"value": "output", "label": "Output method (units delivered, milestones)"},
                        {"value": "input", "label": "Input method (costs incurred, time elapsed)"}
                    ],
                    "next_step": "step_summary"
                },
                "step_summary": {
                    "title": "Summary & Next Steps",
                    "description": "Review your revenue recognition analysis",
                    "instruction": "Review the analysis above. Are there any additional considerations?",
                    "input_type": "text",
                    "required": false,
                    "next_step": null
                },
                "step_no_contract": {
                    "title": "No Contract Identified",
                    "description": "Revenue recognition requires a valid contract",
                    "instruction": "Consider establishing a formal agreement before proceeding with revenue recognition.",
                    "input_type": "text",
                    "required": false,
                    "next_step": null
                }
            }
        }
        
        # Lease Accounting Workflow (ASC 842)
        lease_workflow = {
            "id": "lease_accounting_asc842",
            "name": "Lease Accounting Analysis (ASC 842)",
            "description": "Determine lease classification and accounting treatment under ASC 842",
            "category": "leases",
            "estimated_time": "10-15 minutes",
            "difficulty": "intermediate",
            "start_step": "step1",
            "steps": {
                "step1": {
                    "title": "Lease Identification",
                    "description": "Determine if the contract contains a lease",
                    "instruction": "Does the contract convey the right to control an identified asset for a period of time?",
                    "input_type": "choice",
                    "choices": [
                        {"value": "yes", "label": "Yes - identified asset with right to control"},
                        {"value": "no", "label": "No - service contract only"}
                    ],
                    "conditional_next": {
                        "yes": "step2",
                        "no": "step_no_lease"
                    }
                },
                "step2": {
                    "title": "Lease Term",
                    "description": "Determine the lease term",
                    "instruction": "What is the lease term (including reasonably certain renewal options)?",
                    "input_type": "text",
                    "next_step": "step3",
                    "help_text": "Include periods covered by renewal options that are reasonably certain to be exercised."
                },
                "step3": {
                    "title": "Lease Classification (Lessee)",
                    "description": "Classify the lease as finance or operating",
                    "instruction": "Does any of the following apply?",
                    "input_type": "choice",
                    "choices": [
                        {"value": "finance", "label": "Yes - ownership transfer, purchase option, major part of economic life, or PV ≥ substantially all fair value"},
                        {"value": "operating", "label": "No - none of the finance lease criteria met"}
                    ],
                    "next_step": "step4"
                },
                "step4": {
                    "title": "Initial Measurement",
                    "description": "Calculate initial lease liability and ROU asset",
                    "instruction": "What is the present value of lease payments?",
                    "input_type": "text",
                    "validation": {"type": "number", "min": 0},
                    "next_step": "step_summary",
                    "help_text": "Use the rate implicit in the lease, or if not determinable, your incremental borrowing rate."
                },
                "step_summary": {
                    "title": "Lease Accounting Summary",
                    "description": "Review your lease accounting analysis",
                    "instruction": "Any additional considerations or complexities?",
                    "input_type": "text",
                    "required": false,
                    "next_step": null
                },
                "step_no_lease": {
                    "title": "No Lease Identified",
                    "description": "This appears to be a service contract",
                    "instruction": "Account for this as a service contract under other applicable GAAP.",
                    "input_type": "text",
                    "required": false,
                    "next_step": null
                }
            }
        }
        
        # Save templates
        templates = [revenue_workflow, lease_workflow]
        for template in templates:
            template_file = self.templates_dir / f"{template['id']}.json"
            if not template_file.exists():
                with open(template_file, 'w', encoding='utf-8') as f:
                    json.dump(template, f, indent=2)
                logger.info("Created default template", template_id=template['id'])
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """Get list of available workflow templates"""
        return [
            {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "estimated_time": template.estimated_time,
                "difficulty": template.difficulty
            }
            for template in self.templates.values()
        ]
    
    async def start_workflow(self, template_id: str, session_id: str, user_id: str = "demo-user") -> Dict[str, Any]:
        """Start a new workflow session"""
        try:
            if template_id not in self.templates:
                return {"error": f"Workflow template '{template_id}' not found"}
            
            template = self.templates[template_id]
            workflow_session = WorkflowSession(session_id, template, user_id)
            self.active_sessions[session_id] = workflow_session
            
            # Get first step
            first_step = template.steps[template.start_step]
            
            return {
                "workflow_id": template_id,
                "session_id": session_id,
                "workflow_name": template.name,
                "description": template.description,
                "estimated_time": template.estimated_time,
                "current_step": self._serialize_step(first_step),
                "progress": {
                    "current": 1,
                    "total": len(template.steps),
                    "percentage": round(1 / len(template.steps) * 100, 1)
                }
            }
            
        except Exception as e:
            logger.error("Failed to start workflow", template_id=template_id, error=str(e))
            return {"error": str(e)}
    
    async def process_step_response(
        self, 
        session_id: str, 
        response: str,
        step_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process user response to current workflow step"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Workflow session not found"}
            
            session = self.active_sessions[session_id]
            current_step = session.template.steps[session.current_step]
            
            # Validate response
            validation_result = self._validate_response(current_step, response)
            if not validation_result["valid"]:
                return {
                    "error": validation_result["message"],
                    "current_step": self._serialize_step(current_step)
                }
            
            # Store response
            session.step_data[session.current_step] = response
            session.completed_steps.append(session.current_step)
            session.last_activity = datetime.utcnow()
            
            # Determine next step
            next_step_id = self._get_next_step(current_step, response)
            
            if next_step_id and next_step_id in session.template.steps:
                # Move to next step
                session.current_step = next_step_id
                next_step = session.template.steps[next_step_id]
                
                return {
                    "status": "continue",
                    "current_step": self._serialize_step(next_step),
                    "progress": {
                        "current": len(session.completed_steps) + 1,
                        "total": len(session.template.steps),
                        "percentage": round((len(session.completed_steps) + 1) / len(session.template.steps) * 100, 1)
                    },
                    "previous_responses": session.step_data
                }
            else:
                # Workflow complete
                session.status = 'completed'
                summary = await self._generate_workflow_summary(session)
                
                return {
                    "status": "completed",
                    "summary": summary,
                    "all_responses": session.step_data,
                    "completed_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to process step response", session_id=session_id, error=str(e))
            return {"error": str(e)}
    
    async def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Get current workflow status"""
        if session_id not in self.active_sessions:
            return {"error": "Workflow session not found"}
        
        session = self.active_sessions[session_id]
        current_step = session.template.steps[session.current_step]
        
        return {
            "workflow_name": session.template.name,
            "status": session.status,
            "current_step": self._serialize_step(current_step),
            "progress": {
                "current": len(session.completed_steps) + 1,
                "total": len(session.template.steps),
                "percentage": round((len(session.completed_steps) + 1) / len(session.template.steps) * 100, 1)
            },
            "started_at": session.started_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        }
    
    def _serialize_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Convert WorkflowStep to dictionary"""
        return {
            "id": step.id,
            "title": step.title,
            "description": step.description,
            "instruction": step.instruction,
            "input_type": step.input_type,
            "choices": step.choices,
            "help_text": step.help_text,
            "required": step.required
        }
    
    def _validate_response(self, step: WorkflowStep, response: str) -> Dict[str, Any]:
        """Validate user response against step requirements"""
        if step.required and not response.strip():
            return {"valid": False, "message": "This field is required"}
        
        if step.input_type == "choice" and response:
            valid_choices = [choice["value"] for choice in step.choices]
            if response not in valid_choices:
                return {"valid": False, "message": f"Please select one of: {', '.join(valid_choices)}"}
        
        if step.validation:
            if step.validation.get("type") == "number":
                try:
                    num_value = float(response)
                    if "min" in step.validation and num_value < step.validation["min"]:
                        return {"valid": False, "message": f"Value must be at least {step.validation['min']}"}
                except ValueError:
                    return {"valid": False, "message": "Please enter a valid number"}
        
        return {"valid": True}
    
    def _get_next_step(self, step: WorkflowStep, response: str) -> Optional[str]:
        """Determine next step based on current step and response"""
        if step.conditional_next and response in step.conditional_next:
            return step.conditional_next[response]
        return step.next_step
    
    async def _generate_workflow_summary(self, session: WorkflowSession) -> Dict[str, Any]:
        """Generate summary of completed workflow"""
        try:
            summary = {
                "workflow_name": session.template.name,
                "completed_at": datetime.utcnow().isoformat(),
                "duration_minutes": round((datetime.utcnow() - session.started_at).total_seconds() / 60, 1),
                "steps_completed": len(session.completed_steps),
                "key_decisions": [],
                "recommendations": []
            }
            
            # Extract key decisions based on workflow type
            if session.template.id == "revenue_recognition_asc606":
                summary["recommendations"].extend([
                    "Document your revenue recognition analysis",
                    "Review with your accounting team",
                    "Consider disclosure requirements under ASC 606"
                ])
            elif session.template.id == "lease_accounting_asc842":
                summary["recommendations"].extend([
                    "Document your lease classification analysis",
                    "Set up appropriate journal entries",
                    "Consider disclosure requirements under ASC 842"
                ])
            
            return summary
            
        except Exception as e:
            logger.error("Failed to generate workflow summary", error=str(e))
            return {"error": "Could not generate summary"}