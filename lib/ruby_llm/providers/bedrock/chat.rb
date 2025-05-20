# frozen_string_literal: true

require 'ostruct'

module RubyLLM
  module Providers
    module Bedrock
      # Chat methods for the AWS Bedrock API implementation
      module Chat
        module_function

        def sync_response(connection, payload)
          # Build URL
          base_url = connection.provider.api_base(connection.config)
          base_url = base_url.chomp('/') if base_url.end_with?('/')
          url = completion_url.start_with?('/') ? completion_url : "/#{completion_url}"
          full_url = "#{base_url}#{url}"
          
          # Sign the request
          signature = sign_request(full_url, config: connection.config, payload: payload)
          
          # Prepare headers with AWS signature
          headers = build_headers(signature.headers, streaming: false)
          
          # Prepare body
          body = payload.is_a?(Hash) ? JSON.generate(payload, ascii_only: false) : payload
          
          # Log request (with sensitive information redacted)
          RubyLLM.logger.debug "POST #{full_url}"
          RubyLLM.logger.debug "Headers: #{headers.inspect}"
          
          begin
            response = HTTP.timeout(connection.config.request_timeout)
                          .headers(headers)
                          .post(full_url, body: body)
            
            if response.status.success?
              parsed_body = parse_response_body(response)
              Anthropic::Chat.parse_completion_response(build_response_object(response, parsed_body))
            else
              handle_error_response(response)
            end
          rescue HTTP::Error => e
            raise RubyLLM::Error.new(nil, "HTTP request failed: #{e.message}")
          rescue UnauthorizedError, ForbiddenError => e
            # Provide more helpful error message for AWS auth issues
            if e.message.include?('InvalidSignatureException') || e.message.include?('AccessDeniedException')
              raise UnauthorizedError.new(e.response, 
                "AWS authentication failed. Please check your bedrock_api_key, bedrock_secret_key, and bedrock_region values.")
            else
              raise e
            end
          end
        end

        def format_message(msg)
          if msg.tool_call?
            Anthropic::Tools.format_tool_call(msg)
          elsif msg.tool_result?
            Anthropic::Tools.format_tool_result(msg)
          else
            format_basic_message(msg)
          end
        end

        def format_basic_message(msg)
          {
            role: Anthropic::Chat.convert_role(msg.role),
            content: Media.format_content(msg.content)
          }
        end

        private
        
        def build_response_object(response, parsed_body)
          OpenStruct.new(
            status: response.status.code,
            body: parsed_body,
            headers: response.headers.to_h,
            env: OpenStruct.new(url: response.uri.to_s)
          )
        end
        
        def parse_response_body(response)
          return {} if response.body.to_s.empty?
          
          begin
            JSON.parse(response.body.to_s)
          rescue JSON::ParserError => e
            RubyLLM.logger.error("Failed to parse JSON response: #{e.message}")
            RubyLLM.logger.debug("Response body first 200 chars: #{response.body.to_s[0..200]}")
            raise RubyLLM::Error.new(nil, "Failed to parse response from Bedrock API: #{e.message}")
          end
        end
        
        def handle_error_response(response)
          buffer = response.body.to_s
          
          begin
            error_data = JSON.parse(buffer)
            error_response = OpenStruct.new(
              status: response.status.code,
              body: error_data,
              headers: response.headers.to_h,
              env: OpenStruct.new(url: response.uri.to_s)
            )
            RubyLLM::ErrorHandler.parse_error(provider: self, response: error_response)
          rescue JSON::ParserError
            # When JSON parsing fails, raise a more generic error with the response status
            raise RubyLLM::Error.new(nil, "Failed to parse error response from Bedrock API (HTTP #{response.status.code}): #{buffer[0..200]}")
          end
        end

        def completion_url
          "model/#{@model_id}/invoke"
        end

        def render_payload(messages, tools:, temperature:, model:, stream: false) # rubocop:disable Lint/UnusedMethodArgument
          # Hold model_id in instance variable for use in completion_url and stream_url
          @model_id = model

          system_messages, chat_messages = Anthropic::Chat.separate_messages(messages)
          system_content = Anthropic::Chat.build_system_content(system_messages)

          build_base_payload(chat_messages, temperature, model).tap do |payload|
            Anthropic::Chat.add_optional_fields(payload, system_content:, tools:)
          end
        end

        def build_base_payload(chat_messages, temperature, model)
          {
            anthropic_version: 'bedrock-2023-05-31',
            messages: chat_messages.map { |msg| format_message(msg) },
            temperature: temperature,
            max_tokens: RubyLLM.models.find(model)&.max_tokens || 4096
          }
        end
      end
    end
  end
end