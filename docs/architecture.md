# Multi-Agent Deep Researcher â€” Architecture



## 1. Overview



The Multi-Agent Deep Researcher is an AI research system that uses multiple specialized agents to research a user's question and produce a structured final report.



The system is built around:



* Python

* CrewAI

* Ollama

* LinkUp Search

* Model Context Protocol (MCP)

* FastMCP



The system does not use a traditional RAG pipeline. Instead of retrieving chunks from a vector database, the research agent uses web search to obtain current information.



\---



## 2. High-Level Architecture



```text

&#x20;                        USER

&#x20;                          |

&#x20;                          | Research Query

&#x20;                          v

&#x20;                 +-------------------+

&#x20;                 |    MCP Server     |

&#x20;                 |    FastMCP        |

&#x20;                 +---------+---------+

&#x20;                           |

&#x20;                           v

&#x20;                 +-------------------+

&#x20;                 | Research Service  |

&#x20;                 +---------+---------+

&#x20;                           |

&#x20;                           v

&#x20;                 +-------------------+

&#x20;                 |   Research Crew   |

&#x20;                 |     CrewAI        |

&#x20;                 +---------+---------+

&#x20;                           |

&#x20;             +-------------+-------------+

&#x20;             |             |             |

&#x20;             v             v             v

&#x20;       Search Agent  Analyst Agent  Writer Agent

&#x20;             |             |             |

&#x20;             v             |             |

&#x20;        LinkUp Search      |             |

&#x20;             |             |             |

&#x20;             +-------------+-------------+

&#x20;                           |

&#x20;                           v

&#x20;                  Final Research Report

```



\---



## 3. User Query



The workflow begins when a user provides a research question.



Example:



```text

What are the latest developments in generative AI?

```



The query is passed into the research system.



\---



## 4. MCP Server



The MCP server provides an interface through which an MCP-compatible client can call the research system.



The server exposes a research tool such as:



```text

crew\_research(query)

```



The MCP server does not perform the research itself.



Its responsibility is to:



1\. Receive the user's query.

2\. Pass the query to the research service.

3\. Receive the final research result.

4\. Return the result to the MCP client.



The MCP server is therefore the entry point into the application.



\---



## 5. Research Service



The research service acts as the application-level coordinator.



Its responsibility is to:



1\. Receive the research query.

2\. Create the research crew.

3\. Start the CrewAI workflow.

4\. Wait for the crew to finish.

5\. Return the final result.



Conceptually:



```text

Query

&#x20; |

&#x20; v

Research Service

&#x20; |

&#x20; v

Create Crew

&#x20; |

&#x20; v

Run Crew

&#x20; |

&#x20; v

Return Result

```



\---



## 6. Research Crew



The research crew is responsible for coordinating the agents and tasks.



The crew contains three specialized agents:



```text

+----------------------+

|    Search Agent      |

+----------------------+



+----------------------+

|   Analyst Agent      |

+----------------------+



+----------------------+

|    Writer Agent      |

+----------------------+

```



The agents have different responsibilities.



\---



## 7. Search Agent



The Search Agent is responsible for finding information on the web.



It uses the LinkUp search tool.



Its responsibilities include:



* Understanding what information is required.

* Creating useful search queries.

* Searching the web.

* Collecting relevant sources.

* Returning research material to the workflow.



The search flow is:



```text

Research Question

&#x20;      |

&#x20;      v

Search Agent

&#x20;      |

&#x20;      v

LinkUp Search Tool

&#x20;      |

&#x20;      v

Web

&#x20;      |

&#x20;      v

Search Results

```



The Search Agent should focus primarily on information gathering rather than writing the final report.



\---



## 8. LinkUp Search Tool



The LinkUp Search Tool is the interface between the Search Agent and the LinkUp API.



Conceptually:



```text

Search Agent

&#x20;    |

&#x20;    v

LinkUp Search Tool

&#x20;    |

&#x20;    v

LinkUp API

&#x20;    |

&#x20;    v

Web Search Results

```



The tool will receive information such as:



```text

query

depth

output\_type

```



Possible search depths include:



```text

standard

deep

```



Possible output types include:



```text

searchResults

sourcedAnswer

structured

```



The API key will be stored in an environment variable rather than hard-coded into the source code.



Example:



```text

LINKUP\_API\_KEY=your\_key\_here

```



\---



## 9. Analyst Agent



The Research Analyst receives the information collected by the Search Agent.



Its responsibilities include:



* Understanding the collected information.

* Comparing information from different sources.

* Identifying important findings.

* Removing irrelevant information.

* Organizing the research.

* Identifying useful conclusions.

* Determining what information should appear in the final report.



Conceptually:



```text

Search Results

&#x20;     |

&#x20;     v

Analyst Agent

&#x20;     |

&#x20;     v

Organized Research

```



The Analyst does not primarily perform web searching.



Its main responsibility is reasoning about the research that has already been collected.



\---



## 10. Writer Agent



The Technical Writer receives the analyzed research.



Its responsibility is to transform the research into a readable final report.



The Writer should:



* Organize the report.

* Use appropriate headings.

* Explain important findings.

* Maintain logical flow.

* Include source references when available.

* Avoid unnecessary repetition.

* Produce a professional research document.



Conceptually:



```text

Analyzed Research

&#x20;      |

&#x20;      v

Writer Agent

&#x20;      |

&#x20;      v

Final Research Report

```



\---



## 11. Task Workflow



The agents are connected through tasks.



The planned workflow is:



```text

&#x20;               SEARCH TASK

&#x20;                    |

&#x20;                    v

&#x20;             Search Results

&#x20;                    |

&#x20;                    v

&#x20;              ANALYSIS TASK

&#x20;                    |

&#x20;                    v

&#x20;             Analyzed Research

&#x20;                    |

&#x20;                    v

&#x20;              WRITING TASK

&#x20;                    |

&#x20;                    v

&#x20;            Final Research Report

```



The tasks are executed sequentially.



This means the output of one stage becomes available to the next stage.



\---



## 12. Sequential Execution



The system uses a sequential research pipeline:



```text

Task 1

&#x20; |

&#x20; v

Task 2

&#x20; |

&#x20; v

Task 3

```



Specifically:



```text

Search

&#x20; â†“

Analysis

&#x20; â†“

Writing

```



The Writer does not begin before the required analysis is available.



Likewise, the Analyst receives the output of the search stage.



\---



## 13. Data Flow



The complete data flow is:



```text

USER QUERY

&#x20;   |

&#x20;   v

MCP SERVER

&#x20;   |

&#x20;   v

RESEARCH SERVICE

&#x20;   |

&#x20;   v

RESEARCH CREW

&#x20;   |

&#x20;   v

SEARCH AGENT

&#x20;   |

&#x20;   v

LINKUP SEARCH

&#x20;   |

&#x20;   v

WEB SOURCES

&#x20;   |

&#x20;   v

SEARCH RESULTS

&#x20;   |

&#x20;   v

ANALYST AGENT

&#x20;   |

&#x20;   v

ANALYZED RESEARCH

&#x20;   |

&#x20;   v

WRITER AGENT

&#x20;   |

&#x20;   v

FINAL REPORT

&#x20;   |

&#x20;   v

MCP SERVER

&#x20;   |

&#x20;   v

USER

```



\---



## 14. Configuration



Configuration values should not be hard-coded into the application.



Environment variables will be used for sensitive configuration.



Example:



```text

LINKUP\_API\_KEY=...

```



The local Ollama server will provide the local language model.



The application will communicate with Ollama through its local API.



\---



## 15. Ollama



Ollama provides the local LLM used by the CrewAI agents.



The planned architecture is:



```text

CrewAI

&#x20;  |

&#x20;  v

Ollama

&#x20;  |

&#x20;  v

Local LLM

```



This allows the agents to perform their reasoning using a model running locally.



The model configuration will be isolated from the agent implementation so that the model can be changed later without rewriting the entire application.



\---



## 16. MCP Architecture



MCP provides the external interface for the research application.



The simplified architecture is:



```text

MCP Client

&#x20;    |

&#x20;    | crew\_research(query)

&#x20;    v

MCP Server

&#x20;    |

&#x20;    v

Research Service

&#x20;    |

&#x20;    v

CrewAI

&#x20;    |

&#x20;    v

Agents + Tools

```



The MCP server therefore acts as a bridge between an MCP-compatible client and our research workflow.



\---



## 17. Separation of Responsibilities



The project is divided into separate layers.



### Configuration



Responsible for:



* Environment variables

* Model configuration

* API configuration



### Tools



Responsible for:



* External APIs

* Search functionality



### Agents



Responsible for:



* Reasoning

* Searching

* Analysis

* Writing



### Tasks



Responsible for:



* Defining what each agent must accomplish.



### Crew



Responsible for:



* Connecting agents and tasks.

* Defining execution order.



### Services



Responsible for:



* Application-level orchestration.



### MCP Server



Responsible for:



* Exposing the research system externally.



This separation makes the project easier to understand, test, maintain, and extend.



\---



## 18. Planned Source Structure



```text

src/

â”‚

â”œâ”€â”€ config/

â”‚   â””â”€â”€ settings.py

â”‚

â”œâ”€â”€ tools/

â”‚   â””â”€â”€ search\_tool.py

â”‚

â”œâ”€â”€ agents/

â”‚   â”œâ”€â”€ search\_agent.py

â”‚   â”œâ”€â”€ analyst\_agent.py

â”‚   â””â”€â”€ writer\_agent.py

â”‚

â”œâ”€â”€ tasks/

â”‚   â”œâ”€â”€ search\_task.py

â”‚   â”œâ”€â”€ analysis\_task.py

â”‚   â””â”€â”€ writing\_task.py

â”‚

â”œâ”€â”€ crew/

â”‚   â””â”€â”€ research\_crew.py

â”‚

â”œâ”€â”€ services/

â”‚   â””â”€â”€ research\_service.py

â”‚

â””â”€â”€ server.py

```



\---



## 19. Testing Strategy



The project will be tested layer by layer rather than only testing the complete application at the end.



Planned testing order:



```text

Configuration

&#x20;    â†“

Search Tool

&#x20;    â†“

Search Agent

&#x20;    â†“

Analyst Agent

&#x20;    â†“

Writer Agent

&#x20;    â†“

Tasks

&#x20;    â†“

Crew

&#x20;    â†“

Research Service

&#x20;    â†“

MCP Server

&#x20;    â†“

Complete System

```



This makes debugging much easier because failures can be isolated to individual components.



\---



## 20. Final Architecture



The final system can be summarized as:



```text

&#x20;                        USER

&#x20;                          |

&#x20;                          v

&#x20;                     MCP CLIENT

&#x20;                          |

&#x20;                          v

&#x20;                     MCP SERVER

&#x20;                          |

&#x20;                          v

&#x20;                  RESEARCH SERVICE

&#x20;                          |

&#x20;                          v

&#x20;                   RESEARCH CREW

&#x20;                          |

&#x20;            +-------------+-------------+

&#x20;            |             |             |

&#x20;            v             v             v

&#x20;         SEARCH        ANALYST       WRITER

&#x20;          AGENT         AGENT         AGENT

&#x20;            |

&#x20;            v

&#x20;      LINKUP SEARCH

&#x20;            |

&#x20;            v

&#x20;           WEB

&#x20;            |

&#x20;            v

&#x20;     SEARCH INFORMATION

&#x20;            |

&#x20;            +-------------> ANALYST

&#x20;                             |

&#x20;                             v

&#x20;                      ANALYZED RESEARCH

&#x20;                             |

&#x20;                             v

&#x20;                           WRITER

&#x20;                             |

&#x20;                             v

&#x20;                      FINAL REPORT

```



The architecture is intentionally modular so that individual components can later be replaced or extended without rewriting the entire system.






