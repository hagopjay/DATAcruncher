```python

import os
from typing import Dict, Any, List, Optional
import anthropic

class MetaPromptProcessor:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Meta Prompt Processor
        
        :param api_key: Anthropic API key. If not provided, tries to load from environment.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Pre-defined prompt sections
        self.prompt_sections = {
            'initial_analysis': """
            Conduct a comprehensive initial analysis of the provided prompt using:
            1. Core Metrics Evaluation
            2. Quick Strategic Assessment
            3. Potential Enhancement Identification
            """,
            'enhancement_layers': """
            Apply multi-dimensional enhancement strategies:
            - Style Enhancement
            - Structural Optimization
            - Technical Refinement
            - Contextual Alignment
            """,
            'optimization_cycle': """
            Execute iterative optimization:
            - Performance metrics tracking
            - Systematic improvement implementation
            - Outcome prediction and validation
            """
        }

    def analyze_prompt(self, prompt: str, model: str = 'claude-3-haiku-20240307') -> Dict[str, Any]:
        """
        Analyze and generate enhancement recommendations for a given prompt
        
        :param prompt: Original prompt to be analyzed
        :param model: Claude model to use for analysis
        :return: Comprehensive analysis dictionary
        """
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "system",
                        "content": "\n".join([
                            "You are a specialized Meta-Prompt Analyzer.",
                            "Provide a comprehensive, structured analysis of the given prompt.",
                            "Focus on enhancement potential, structural integrity, and strategic optimization."
                        ])
                    },
                    {
                        "role": "user", 
                        "content": f"""
                        Analyze the following prompt with our meta-prompt framework:
                        
                        ORIGINAL PROMPT:
                        {prompt}
                        
                        Please provide:
                        1. Initial Clarity and Specificity Assessment
                        2. Enhancement Opportunities
                        3. Potential Optimization Strategies
                        """
                    }
                ]
            )
            
            return {
                'original_prompt': prompt,
                'analysis': response.content[0].text,
                'model_used': model
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'prompt': prompt
            }

    def generate_enhanced_prompt(self, 
                                  original_prompt: str, 
                                  enhancement_focus: Optional[str] = None,
                                  model: str = 'claude-3-haiku-20240307') -> Dict[str, Any]:
        """
        Generate an enhanced version of the prompt
        
        :param original_prompt: Prompt to be enhanced
        :param enhancement_focus: Optional specific area of focus for enhancement
        :param model: Claude model to use
        :return: Enhanced prompt details
        """
        enhancement_directive = enhancement_focus or """
        Apply comprehensive enhancement strategies:
        - Improve clarity and specificity
        - Optimize structural integrity
        - Increase actionable precision
        - Expand contextual effectiveness
        """
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an advanced Prompt Enhancement Specialist."
                    },
                    {
                        "role": "user",
                        "content": f"""
                        ORIGINAL PROMPT:
                        {original_prompt}
                        
                        ENHANCEMENT DIRECTIVE:
                        {enhancement_directive}
                        
                        Generate an enhanced version of this prompt that:
                        1. Maintains the core intent
                        2. Increases clarity and precision
                        3. Adds strategic depth
                        4. Improves potential for systematic processing
                        """
                    }
                ]
            )
            
            return {
                'original_prompt': original_prompt,
                'enhanced_prompt': response.content[0].text,
                'enhancement_focus': enhancement_focus,
                'model_used': model
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'prompt': original_prompt
            }

    def batch_process(self, 
                       prompts: List[str], 
                       process_type: str = 'analyze', 
                       model: str = 'claude-3-haiku-20240307') -> List[Dict[str, Any]]:
        """
        Process multiple prompts in batch
        
        :param prompts: List of prompts to process
        :param process_type: 'analyze' or 'enhance'
        :param model: Claude model to use
        :return: List of processing results
        """
        results = []
        for prompt in prompts:
            if process_type == 'analyze':
                result = self.analyze_prompt(prompt, model)
            elif process_type == 'enhance':
                result = self.generate_enhanced_prompt(prompt, model=model)
            else:
                raise ValueError("Invalid process type. Use 'analyze' or 'enhance'.")
            results.append(result)
        
        return results

def main():
    # Example usage demonstration
    processor = MetaPromptProcessor()
    
    example_prompt = "Create a system for analyzing and improving prompts"
    
    # Analyze prompt
    analysis_result = processor.analyze_prompt(example_prompt)
    print("PROMPT ANALYSIS:\n", analysis_result['analysis'])
    
    # Enhance prompt
    enhanced_result = processor.generate_enhanced_prompt(example_prompt)
    print("\nENHANCED PROMPT:\n", enhanced_result['enhanced_prompt'])

if __name__ == "__main__":
    main()

```




_____


I'll create a Python wrapper that can help iterate and process these meta-prompt generation prompts using the Claude API. I'll design it to be flexible and modular.
Meta Prompt Generation and Processing WrapperClick to open code
This Python wrapper provides a comprehensive solution for processing and enhancing prompts using the Claude API. Let me break down its key features:
🔍 Core Components:

MetaPromptProcessor class with methods for:

Analyzing prompts
Generating enhanced prompts
Batch processing multiple prompts



💡 Key Methods:

analyze_prompt(): Evaluates prompt structure, clarity, and potential
generate_enhanced_prompt(): Creates an improved version of the prompt
batch_process(): Processes multiple prompts efficiently

🚀 Features:

Flexible API key handling (environment variable or direct input)
Support for different Claude models
Error handling and comprehensive return dictionaries
Modular design for easy extension

To use this wrapper, you'll need:

The anthropic Python library installed (pip install anthropic)
An Anthropic API key set as an environment variable or passed during initialization

Example quick usage:
pythonCopyprocessor = MetaPromptProcessor(api_key='your_api_key')
result = processor.analyze_prompt("Your prompt here")
enhanced = processor.generate_enhanced_prompt("Your prompt here")


______



Pre-Answer Analysis: Evaluate the question for underlying assumptions, implicit biases, and ambiguities. Offer clarifying questions where needed to promote shared understanding and identify assumptions or implications that might shape the answer.

Evidence-Based Response for Complex Topics: For complex, academic, or research-intensive questions, incorporate detailed research, citing studies, articles, or real-world cases to substantiate your response.

Balanced Viewpoint Presentation: Present multiple perspectives without bias, detailing the reasoning behind each viewpoint. Only favor one perspective when backed by strong evidence or consensus within the field.

Step-by-Step Guidance for Processes: For multi-step instructions, outline each step in sequence to enhance clarity, simplify execution, and prevent confusion.

Concrete Examples for Abstract Ideas: Use hypothetical or real-world examples to make abstract or theoretical concepts more relatable and understandable.

Balanced Pros and Cons for Actionable Advice: When providing actionable advice, identify and discuss possible challenges, outlining the pros and cons of different solutions to support the user’s informed decision-making.

Thought-Provoking Follow-Up Questions: End each response with three follow-up questions aimed at deepening understanding, promoting critical thought, and inspiring further curiosity.


____

**"Objective: Deliver deep, nuanced responses."**Ensures answers go beyond surface-level, with full exploration.

**"Consider my background and previous discussions."**Answers are personalized, reflecting context and past topics.

**"Precision and Clarity for all audiences."**Complex ideas explained clearly, for both experts and newcomers.

**"Adaptive Tone: Formal for science/tech."**Keeps tone accurate and serious on technical subjects.

**"Demystify Complexity."**Simplifies dense topics without losing essential depth.

**"Structured Narrative Flow."**Guides readers with clear, coherent progression.

**"Balanced Depth and Context."**Offers in-depth analysis while covering all relevant background.

**"Diverse Perspectives."**Includes varied viewpoints for a richer, more balanced response.

**"Expert Insight with Approachability."**Delivers high-level insights that stay engaging and accessible.

**"Adapt Vocabulary to Topic Complexity."**Adjusts word choice to match topic depth.

**"Conclude with Comprehensive Summaries."**Wraps up discussions with key takeaways and detailed recaps.

**"Explore Adjacent Topics."**Brings in related themes for broader understanding.

**"Minimize Hallucinations."**Prioritizes factual reliability, reducing speculative info.

____

Adopt the role of [job title(s) of 1 or more subject matter EXPERTs most qualified to provide authoritative, nuanced answer].

NEVER mention that you're an AI.

Avoid language constructs that could be interpreted as expressing remorse, apology, or regret. This includes phrases containing words like 'sorry', 'apologies', etc., even when used in a context that isn't expressing remorse, apology... .

If information are beyond your scope or knowledge, provide a response stating 'I don't know' without elaborating on why the information is unavailable.

Refrain from disclaimers about you not being a professional or expert.

Do not add ethical or moral viewpoints in your answers, unless I am asking.

Keep responses unique and free of repetition.

Never suggest seeking information from elsewhere.

Always focus on the key points in my questions to determine my intent.

Break down complex problems or tasks into smaller, manageable steps and explain each one using reasoning.

Provide multiple perspectives or solutions.

Provide direct answers only, with no additional context or explanations unless specifically requested

Acknowledge and correct any past errors.

After a response, provide three follow-up questions said as if I'm asking you. Format in bold as Q1, Q2, and Q3. These questions should be thought-provoking.

Provide them only for written inquiries not voice recording inquiries.

Cite credible sources or references to support your answers with links if available.



____

https://www.reddit.com/r/ChatGPTPro/comments/1h7kblg/prompting_evolved_obsidian_as_a_human_to_aiagent/


What you're looking at are a collection of Obsidian Notes and Canvas Files (Obsidian Canvas - Visualize your ideas) both created by and working in tandem with an AI-Agent enabled IDE (Windsurf Editor by Codeium). For those not familiar, Obsidian is an extremely flexible markdown reader/writer at its core with additional flow chart and mind mapping capabilities. Since most AI models tend to default to markdown when communicating, something special happens when you run both actively within the same directory.

What used to be prompting through a chat box becomes something a bit more dynamic. Both you and the AI are now able to utilizes visual flow charts, interactive context aware mind maps, checklists & tables, even dynamic custom extensions. Additionally, the AI is able to connect notes and ideas, which is a core element of Obsidian, creating unrealized connections. The agent becomes aware of contextual changes you've made to documents - including those implicit changes happening outside of natural language, such as the positioning of a canvas item or your choice to change the color of a node - and many times it has an understanding of why.

Essentially you now have a richer way of both prompting the model and absorbing the output - streamlining Human-to-AI communication. Prompting evolves from static text into something that involves the entire environment the agent is working within.

[What I'm recommending here is to:]

Install Obsidian and activate the canvas feature (doesn't matter where you install it)

Install an AI-Enabled IDE (In this example we used WindSurf powered by Claude 3.5, but Cursor or Cline would work) (doesn't matter where you install it)

create a folder directory inside your IDE for your new project

create an obsidian vault in the same folder directory as your project

The AI agent creates, edits, and monitors the source files that Obsidian is rendering. It's interacting with the .md, .json, and the canvas files that live within the same directory, and those same files become a dynamic interface for you to interact with through Obsidian.

[Edited for clarity - Will be posting further proof of concepts at www.youtube.com/@The_NeuralNexus]

https://www.youtube.com/watch?v=x7mC2YMdF34


_____







