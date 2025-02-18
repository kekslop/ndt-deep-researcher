language_detection_instructions = """You are a language detection expert.
Your task is to detect the language of the provided text accurately.

<TEXT>
{input_text}
</TEXT>

Determine the natural language of the text and provide details about it."""

query_writer_instructions = """Your goal is to generate a targeted web search query in the same language as the user's input.
The query will gather information related to a specific topic.

<TOPIC>
{research_topic}
</TOPIC>

<EXAMPLE>
Example for English input:
- Input: "Tell me about machine learning"
- Response would include "machine learning transformer architecture explained" as a query
</EXAMPLE>"""

summarizer_instructions = """
<GOAL>
Generate a high-quality summary of the web search results and keep it concise / related to the user topic.
Respond in the same language as the user's original query.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully                                                    
2. Compare the new information with the existing summary                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition                            
    c. If it's not relevant to the user topic, skip it                                                            
4. Ensure all additions are relevant to the user's topic                                                         
5. Verify that your final output differs from the input summary"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.
Answer in the same language as the summary.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for web search.</REQUIREMENTS>"""