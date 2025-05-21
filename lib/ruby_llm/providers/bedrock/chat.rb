# frozen_string_literal: true

module RubyLLM
  module Providers
    module Bedrock
      # Chat methods for the AWS Bedrock API implementation using the official SDK
      module Chat
        module_function

        # Process a synchronous (non-streaming) chat completion request
        # @param connection [RubyLLM::Connection] The connection object
        # @param payload [Hash] The request payload to send
        # @return [RubyLLM::Message] The model's response message
        def sync_response(connection, payload)
          client = bedrock_runtime_client(connection.config)
          
          begin
            response = client.invoke_model({
              model_id: @model_id,
              body: JSON.generate(payload, ascii_only: false)
            })
            
            # Parse the response body
            parsed_response = JSON.parse(response.body.read)
            
            # Use anthropic's parsing logic for the response
            Anthropic::Chat.parse_completion_response_body(parsed_response)
          rescue Aws::Errors::ServiceError, Aws::Errors::NonHttpError => e
            handle_aws_error(e)
          rescue JSON::ParserError => e
            raise Error::ResponseError.new(
              provider: slug,
              message: "Failed to parse Bedrock response: #{e.message}"
            )
          end
        end

        # Format a message for the Bedrock API
        # @param msg [RubyLLM::Message] The message to format
        # @return [Hash] Formatted message for the Bedrock API
        def format_message(msg)
          if msg.tool_call?
            Anthropic::Tools.format_tool_call(msg)
          elsif msg.tool_result?
            Anthropic::Tools.format_tool_result(msg)
          else
            format_basic_message(msg)
          end
        end

        # Format a basic message (not tool-related) for the Bedrock API
        # @param msg [RubyLLM::Message] The message to format
        # @return [Hash] Formatted basic message for the Bedrock API
        def format_basic_message(msg)
          {
            role: Anthropic::Chat.convert_role(msg.role),
            content: Media.format_content(msg.content)
          }
        end

        private

        # Render the payload for a Bedrock API request
        # @param messages [Array<RubyLLM::Message>] The messages to send
        # @param tools [Hash] The tools to make available
        # @param temperature [Float] The temperature setting
        # @param model [String] The model ID to use
        # @param stream [Boolean] Whether to stream the response
        # @return [Hash] The formatted payload
        def render_payload(messages, tools: nil, temperature:, model:, stream: false) # rubocop:disable Lint/UnusedMethodArgument
          # Hold model_id in instance variable for use in request
          @model_id = model

          system_messages, chat_messages = Anthropic::Chat.separate_messages(messages)
          system_content = Anthropic::Chat.build_system_content(system_messages)

          build_base_payload(chat_messages, temperature, model).tap do |payload|
            # Add system content
            payload[:system] = system_content unless system_content.empty?
            
            # Add tools if provided
            if tools && !tools.empty?
              # Format tools specifically for Bedrock
              formatted_tools = tools.values.map { |t| function_for_bedrock(t) }
              payload[:tools] = formatted_tools
              
              # Debug tool formatting
              RubyLLM.logger.debug { "Bedrock tools payload: #{formatted_tools.inspect}" }
              
              # NOT using tool_choice since it may not be supported by all versions
            end
          end
        end

        # Build the base payload for a Bedrock API request
        # @param chat_messages [Array<RubyLLM::Message>] The chat messages to send
        # @param temperature [Float] The temperature setting
        # @param model [String] The model ID to use
        # @return [Hash] The base payload
        def build_base_payload(chat_messages, temperature, model)
          {
            anthropic_version: DEFAULTS[:anthropic_version],
            messages: chat_messages.map { |msg| format_message(msg) },
            temperature: temperature,
            max_tokens: RubyLLM.models.find(model)&.max_tokens || 4096
          }
        end
        
        # Format a tool specifically for Bedrock API, which has different requirements than Anthropic
        # @param tool [Object] The tool to format
        # @return [Hash] A tool definition in Bedrock's format
        def function_for_bedrock(tool)
          # Extract parameters in the format Bedrock expects
          properties = {}
          required = []
          
          if tool.respond_to?(:parameters) && tool.parameters.is_a?(Hash)
            tool.parameters.each do |name, param|
              name_str = name.to_s
              
              # Extract property definition
              properties[name_str] = {
                "type" => param.respond_to?(:type) ? param.type.to_s : "string",
                "description" => param.respond_to?(:description) ? param.description.to_s : ""
              }
              
              # Track required parameters
              required << name_str if param.respond_to?(:required) && param.required
            end
          end
          
          # For non-streaming requests, we use invoke_model which just needs JSON payload
          # Return tool definition in Bedrock's preferred format for standard JSON payload
          {
            name: tool.name.to_s,
            description: tool.description.to_s,
            input_schema: {
              type: "object",
              properties: properties,
              required: required
            }
          }
        end
      end
    end
  end
end