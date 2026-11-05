---
layout: default
title: Tool Search
parent: "Tools"
nav_order: 5
description: Give a model hundreds of tools without sending them all, and let it search for the ones it needs.
redirect_from:
  - /guides/tool-search
---

# {{ page.title }}
{: .no_toc }

{{ page.description }}
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

After reading this guide, you will know:

*   Why large tool sets need tool search.
*   How to defer the tools of an MCP server, a chat, an agent, or a tool class.
*   How to use your own search implementation.
*   How to see which tools are deferred and which ones the model found.
*   Which providers search for tools, and what happens on the others.

## Why Defer Tools

Every tool a chat registers sends its name, description, and schema with every request. An MCP server like GitHub's offers dozens of tools, and connecting a few servers puts hundreds of definitions in front of the model before it reads your question. That costs input tokens on every turn, and models pick the right tool less often when they have to choose from a long list.

A deferred tool stays out of the model's context. The provider gives the model a search tool instead, and the model searches for what it needs: "weather" finds `weather_lookup`. The provider then loads the matching definitions, and the model calls them like any other tool. Your tools run the same way they always do.

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

## Search tools in your application

Declare `searches_tools` on a Ruby tool to use your own search index or ranking logic:

```ruby
class FindTools < RubyLLM::Tool
  description "Finds tools for the task"
  searches_tools

  def execute(query:)
    RubyLLM::ToolSearchResults.new(*Catalog.search(query))
  end
end

chat.with_tools(FindTools)
chat.with_tools(*Catalog.tools, defer: true)
```

`Catalog` is your application's tool catalog. Its `search` method returns matching tool classes, tool instances, or model-visible names. `ToolSearchResults` accepts all three, preserves their order, and removes duplicates. Return `RubyLLM::ToolSearchResults.new` when nothing matches.

Register each matching tool with the chat before returning it. You can return eager tools as well as deferred ones. For MCP tools, return the registered name, including any prefix or alias. Hidden tools, unavailable names, and the search tool itself raise `ArgumentError` before RubyLLM stores the result. Correct the search result and call `run_tools` or `complete` again to retry the pending call.

Register one search tool per chat. Search tools are always eager, even when their class declares `defer` or you register them with `defer: true`:

```ruby
class SupportAgent < RubyLLM::Agent
  model "{{ site.models.anthropic_current }}"
  tools FindTools, *Catalog.tools, defer: true
end
```

Search calls use the ordinary tool loop, including callbacks, approvals, and cancellation. Return an error Hash for a recoverable search failure, as with any other tool. An MCP tool cannot be the search handler, but your Ruby search tool can return MCP tool names.

Anthropic and OpenAI Responses load the returned definitions through their native tool search formats. Other protocols receive an ordinary function call and JSON results, with every registered tool offered up front. OpenAI Responses cannot force a named search handler through `with_tool_options(choice: :find_tools)` because its search declaration has no name. Use `:auto` or `:required` instead.

With application search on OpenAI Responses, deferred tools stay out of the request's tool definitions. Search results load their definitions where discovery occurs in the conversation. Hosted search still receives the full catalog.

Search results persist as tool names in ordinary message content. When you reload a Rails chat, register its tools again through `with_tools` or its Agent. RubyLLM renders the current definitions. If an old result names a tool you have since removed or hidden, RubyLLM omits that match and logs a warning without changing the stored result.

Removing the search handler makes earlier calls replay as ordinary function calls and results. Replacing or renaming it preserves native OpenAI search history. Batch results do not expose client search calls as executable Ruby tool calls.

## Agents and Rails

Agents pass `defer:` from their `tools` and `mcp` macros:

```ruby
class SupportAgent < RubyLLM::Agent
  model "{{ site.models.anthropic_current }}"
  tools LookupAccount
  mcp Zendesk, Stripe, defer: true
end
```

Chat records respond to `with_tools` and `with_mcp` with `defer:`, and agents with a `chat_model` apply their deferred tools to the records they create and find. The search a model ran is part of its reply, so a persisted chat that keeps `raw_content` on its messages replays it on the next turn like any other reply.

## Reading Deferred Tools

`deferred_tools` returns the deferred tools, keyed by name like `tools`:

```ruby
chat.tools.keys           # => [:current_time, :search_issues, :create_issue, ...]
chat.deferred_tools.keys  # => [:search_issues, :create_issue, ...]
```

A provider-hosted search appears on the reply that ran it, in `server_tool_calls`, before the tool calls it made possible:

```ruby
response = chat.messages.find(&:tool_call?)
response.server_tool_calls.any?  # => true
response.tool_calls.values.map(&:name)  # => ["search_issues"]
```

Your own search handler appears in `tool_calls` and its results appear as tool messages, like any other Ruby tool.

A tool the model found stays found for the rest of the conversation. RubyLLM sends the same tool list on every turn, so a prompt cache that covers the tools keeps hitting.

## Choosing the Search Tool

Without an application search handler, RubyLLM adds the provider's default search tool. Anthropic also offers a regular-expression search. Enable it with `with_provider_tools`, and RubyLLM uses it instead of adding the default:

```ruby
chat.with_provider_tools({ type: "tool_search_tool_regex_20251119", name: "tool_search_tool_regex" })
```

See [Provider Tools]({% link _core_features/provider-tools.md %}) for how raw tool definitions work.

## Provider Support

RubyLLM defers tools wherever it speaks Anthropic's Messages API or OpenAI's Responses API, including Claude on Vertex AI. It does not check the model first: a model without tool search answers with the provider's error. Check the provider's documentation for the models that support it.

Other providers and protocols, such as Gemini and OpenAI's Chat Completions API, receive deferred tools as ordinary tools. The same code runs everywhere, and only the token savings depend on the provider.

When a chat switches to another model of the same provider, the tools the earlier model found stay found. Provider-hosted search history starts over when you switch providers. Application search results remain in the local transcript and render through the selected protocol. Protocols without tool search receive every tool.

## Next Steps

*   [Tools]({% link _core_features/tools.md %})
*   [Model Context Protocol]({% link _core_features/mcp.md %})
*   [Provider Tools]({% link _core_features/provider-tools.md %})
*   [Anthropic tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
*   [OpenAI tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)
