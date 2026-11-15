---
layout: default
title: Tool Search
nav_order: 5
description: Give a model hundreds of tools without sending them all, and let it search for the ones it needs.
redirect_from:
  - /guides/tool-search
---

# {{ page.title }}
{: .no_toc }

{{ page.description }}
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

After reading this guide, you will know:

*   Why large tool sets need tool search.
*   How to defer the tools of an MCP server, a chat, an agent, or a tool class.
*   How to see which tools are deferred and which ones the model found.
*   How to search with a tool of your own.
*   Which providers search for tools, and what happens on the others.

## Why Defer Tools

Every tool a chat registers sends its name, description, and schema with every request. An MCP server like GitHub's offers dozens of tools, and connecting a few servers puts hundreds of definitions in front of the model before it reads your question. That costs input tokens on every turn, and models pick the right tool less often when they have to choose from a long list.

A deferred tool stays out of the model's context. The model gets a search tool instead, and searches for what it needs: "weather" finds `weather_lookup`. The matching definitions are then loaded, and the model calls them like any other tool. Your tools run the same way they always do.

## Deferring an MCP Server's Tools

MCP servers are where tool lists grow longest. Pass `defer: true` to `with_mcp`:

```ruby
chat = RubyLLM.chat(model: "{{ site.models.anthropic_current }}")
chat.with_mcp(GitHub.new(user: current_user), Linear.new(user: current_user), defer: true)

chat.ask "Which open issues mention the flaky login spec?"
```

To defer some of a server's tools every time you connect it, declare them in the class. Without names, `defer` defers every tool of the server:

```ruby
class GitHub < RubyLLM::MCP
  url "https://api.githubcopilot.com/mcp/"
  bearer_token ENV.fetch("GITHUB_TOKEN")

  defer :search_code, :list_workflows, :get_workflow_run
end
```

`defer: false` on `with_mcp` offers every tool of the server up front in one chat, whatever the class declares. Tool classes you add to a server with `tool` keep their own `defer`.

## Deferring Tools in a Chat

`with_tools` takes the same option:

```ruby
chat.with_tools(CurrentTime)
chat.with_tools(*Catalog.tools, defer: true)
```

`CurrentTime` stays in front of the model on every turn, and the catalog's tools wait to be found. Keep the tools a conversation almost always needs undeferred, so the model can call them without searching first.

A tool that should always wait to be found declares it in its class:

```ruby
class DeepResearch < RubyLLM::Tool
  description "Runs a multi-step web investigation"
  defer

  parameter :query, description: "What to investigate"

  def execute(query:)
    Research.run(query)
  end
end

chat.with_tools(DeepResearch)
chat.with_tools(DeepResearch, defer: false)
```

The second call registers it as an ordinary tool for that chat.

To defer one tool and not others of its class, call `defer` on it:

```ruby
chat.with_tools(CurrentTime.new, *connector_tools.map(&:defer))
```

## Agents and Rails

Agents pass `defer:` from their `tools` and `mcp` macros:

```ruby
class SupportAgent < RubyLLM::Agent
  model "{{ site.models.anthropic_current }}"
  tools LookupAccount
  mcp Zendesk, Stripe, defer: true
end
```

A lambda decides when the chat is built, so a chat with few tools can keep them all in front of the model:

```ruby
class SupportAgent < RubyLLM::Agent
  inputs :account
  mcp(defer: -> { account.connectors.many? }) { account.connectors.map(&:server) }
end
```

Chat records respond to `with_tools` and `with_mcp` with `defer:`, and agents with a `chat_class` apply their deferred tools to the records they create and find. The search a model ran is part of its reply, so a persisted chat that keeps `raw_content` on its messages replays it on the next turn like any other reply.

## Reading Deferred Tools

`deferred_tools` returns the deferred tools, keyed by name like `tools`:

```ruby
chat.tools.keys           # => [:current_time, :search_issues, :create_issue, ...]
chat.deferred_tools.keys  # => [:search_issues, :create_issue, ...]
```

The search appears on the reply that ran it, in `server_tool_calls`, before the tool calls it made possible:

```ruby
response = chat.messages.find(&:tool_call?)
response.server_tool_calls.any?  # => true
response.tool_calls.map(&:name)  # => ["search_issues"]
```

A tool the model found stays found for the rest of the conversation. RubyLLM sends the same tool list on every turn, so a prompt cache that covers the tools keeps hitting.

## Choosing the Search Tool

RubyLLM adds the provider's default search tool. Anthropic also offers a regular-expression search. Enable it with `with_provider_tools`, and RubyLLM uses it instead of adding the default:

```ruby
chat.with_provider_tools({ type: "tool_search_tool_regex_20251119", name: "tool_search_tool_regex" })
```

See [Provider Tools]({% link _tools/provider-tools.md %}) for how raw tool definitions work.

## Searching With Your Own Tool

The provider's search ranks your tools by their names and descriptions. When your application has a better index, such as embeddings, or a catalog scoped to the current account, write the search as a tool and declare `searches_tools`:

```ruby
class CatalogSearch < RubyLLM::Tool
  description "Finds tools for a task. Describe the task in a few words."
  searches_tools

  def execute(query:)
    RubyLLM::ToolSearchResults.new(*ToolIndex.nearest(query, limit: 5))
  end
end

chat = RubyLLM.chat(model: "{{ site.models.anthropic_current }}")
chat.with_tools(CatalogSearch, *Catalog.tools, defer: true)
chat.ask "Refund order 42"
```

`searches_tools` makes `CatalogSearch` the chat's search tool, on every provider. The model calls it `tool_search`, whatever the Ruby class is named. A declared search tool cannot have a namespace or override that name. Naming an ordinary tool `tool_search` does not make it a search tool.

The search is never deferred, even inside `defer: true`, and the chat offers it while it has deferred tools. `ToolSearchResults.new` takes Tool classes, Tool instances, or the names the chat registers tools under (`:crm_customer_lookup` for a namespaced one), and the chat loads the deferred tools it names. A name that is not one of the chat's deferred tools loads nothing. Return a String instead when nothing matched and the model should hear why.

Claude and GPT load the tools your search found the way they load the ones their own search finds, so the tool list stays the same across loads and a prompt cache keeps hitting. Every other provider gets the definitions appended to the next request, as with `tool_search`. The search and its result are an ordinary tool call in `messages`, so a persisted chat replays them on any provider, and what the search loaded stays loaded when the chat moves to another provider.

Agents declare the search tool like any other:

```ruby
class SupportAgent < RubyLLM::Agent
  model "{{ site.models.openai_current }}"
  tools ConnectorSearch
  mcp Zendesk, Stripe, defer: true
end
```

Registering another tool declared with `searches_tools` replaces the chat's search implementation. It keeps the `tool_search` identity, so earlier search results still load their tools. An ordinary tool named `tool_search` cannot be registered in the same chat as a declared search tool; either registration order raises `ArgumentError`.

An MCP server's tool cannot declare `searches_tools`, so a search that runs on a server is a Tool of your own that calls it.

## Provider Support

RubyLLM uses the provider's own tool search wherever it speaks Anthropic's Messages API or OpenAI's Responses API, including Claude on Bedrock and Vertex AI and GPT on Azure. It does not check the model first: a model without tool search answers with the provider's error. Check the provider's documentation for the models that support it. xAI is the exception: it speaks the Responses API, but limits its tool search to alpha accounts, so xAI chats get RubyLLM's own search.

Every other provider and protocol, such as Gemini, xAI, Bedrock's Converse API (which Claude uses only with `protocol: :converse`), and OpenAI's Chat Completions API, gets a `tool_search` tool from RubyLLM instead. Its description lists the deferred tools by name, and the model calls it with a few words, or with `select:` and tool names:

```ruby
chat = RubyLLM.chat(model: "{{ site.models.gemini_current }}")
chat.with_tools(CurrentTime).with_tools(*Catalog.tools, defer: true)

chat.tools.keys  # => [:current_time, :search_issues, ..., :tool_search]
```

The tools it finds are sent with every request after that, after the tools that are not deferred, in the order the conversation first loaded or called them. A deferred tool the conversation called stays loaded too. Each request repeats the one before it and adds to its end, so a prompt cache that covers the tools keeps hitting until the model loads another one. The search and its result are an ordinary tool call in `messages`, so a persisted chat keeps what it loaded.

When a chat switches to another model of the same provider, the tools the earlier model found stay found. Switching to another provider starts over: a model with tool search searches again with the provider's search, and one without it loads tools with `tool_search`.

## Next Steps

*   [Tools]({% link _getting_started/tools.md %})
*   [Model Context Protocol]({% link _tools/mcp.md %})
*   [Provider Tools]({% link _tools/provider-tools.md %})
*   [Anthropic tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
*   [OpenAI tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)
