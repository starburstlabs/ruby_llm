# frozen_string_literal: true

module RubyLLM
  module Providers
    module Anthropic
      # Chat methods of the OpenAI API integration
      module Chat
        module_function

        def completion_url
          '/v1/messages'
        end

        def render_payload(messages, tools:, temperature:, model:, stream: false)
          system_messages, chat_messages = separate_messages(messages)
          system_content = build_system_content(system_messages)

          build_base_payload(chat_messages, temperature, model, stream).tap do |payload|
            add_optional_fields(payload, system_content:, tools:)
          end
        end

        def separate_messages(messages)
          messages.partition { |msg| msg.role == :system }
        end

        def build_system_content(system_messages)
          if system_messages.length > 1
            RubyLLM.logger.warn(
              "Anthropic's Claude implementation only supports a single system message. " \
              'Multiple system messages will be combined into one.'
            )
          end

          system_messages.map { |msg| format_message(msg)[:content] }.join("\n\n")
        end

        def build_base_payload(chat_messages, temperature, model, stream)
          {
            model: model,
            messages: chat_messages.map { |msg| format_message(msg) },
            temperature: temperature,
            stream: stream,
            max_tokens: RubyLLM.models.find(model)&.max_tokens || 4096
          }
        end

        def add_optional_fields(payload, system_content:, tools:)
          payload[:tools] = tools.values.map { |t| Tools.function_for(t) } if tools.any?
          payload[:system] = system_content unless system_content.empty?
        end

        def parse_completion_response(response)
          # Special handling for HTTP.rb responses
          RubyLLM.logger.debug("Response type: #{response.class}")
          if response.is_a?(HTTP::Response)
            response_body = response.body.to_s
            RubyLLM.logger.debug("HTTP::Response body: #{response_body[0..200]}")
          elsif response.respond_to?(:body) && response.body.is_a?(Hash)
            # Body is already a hash (parsed JSON)
            data = response.body
            RubyLLM.logger.debug("Response body is already a hash")
          elsif response.respond_to?(:body)
            response_body = response.body.to_s
            RubyLLM.logger.debug("Response body: #{response_body[0..200]}")
          else
            raise "Unexpected response type: #{response.class}"
          end

          # Parse JSON if needed
          data = if defined?(response_body) && response_body
                   begin
                     JSON.parse(response_body)
                   rescue JSON::ParserError => e
                     RubyLLM.logger.error("Failed to parse response JSON: #{e.message}")
                     RubyLLM.logger.debug("Response body: #{response_body ? response_body[0..500] : 'nil'}")
                     raise RubyLLM::Error.new(nil, "Failed to parse response JSON: #{e.message}")
                   end
                 else
                   # If we don't have a response_body, we should already have data from a pre-parsed body
                   data || {}
                 end
                 
          content_blocks = data['content'] || []

          text_content = extract_text_content(content_blocks)
          tool_use = Tools.find_tool_use(content_blocks)

          build_message(data, text_content, tool_use)
        end

        def extract_text_content(blocks)
          return "" if blocks.is_a?(String)
          text_blocks = blocks.select { |c| c['type'] == 'text' }
          text_blocks.map { |c| c['text'] }.join
        end

        def build_message(data, content, tool_use)
          Message.new(
            role: :assistant,
            content: content,
            tool_calls: Tools.parse_tool_calls(tool_use),
            input_tokens: data.dig('usage', 'input_tokens'),
            output_tokens: data.dig('usage', 'output_tokens'),
            model_id: data['model']
          )
        end

        def format_message(msg)
          if msg.tool_call?
            Tools.format_tool_call(msg)
          elsif msg.tool_result?
            Tools.format_tool_result(msg)
          else
            format_basic_message(msg)
          end
        end

        def format_basic_message(msg)
          {
            role: convert_role(msg.role),
            content: Media.format_content(msg.content)
          }
        end

        def convert_role(role)
          case role
          when :tool, :user then 'user'
          else 'assistant'
          end
        end
      end
    end
  end
end