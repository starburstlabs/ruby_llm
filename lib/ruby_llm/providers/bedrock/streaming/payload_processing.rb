# frozen_string_literal: true

require 'base64'

module RubyLLM
  module Providers
    module Bedrock
      module Streaming
        # Module for processing payloads from AWS Bedrock streaming responses.
        # Handles JSON payload extraction, decoding, and chunk creation.
        #
        # Responsibilities:
        # - Extracting and validating JSON payloads
        # - Decoding Base64-encoded response data
        # - Creating response chunks from processed data
        # - Error handling for JSON parsing and processing
        #
        # @example Processing a payload
        #   process_payload(raw_payload) do |chunk|
        #     yield_chunk_to_client(chunk)
        #   end
        module PayloadProcessing
          def process_payload(payload, &)
            RubyLLM.logger.debug("Processing payload: #{payload[0..100]}...")
            json_payload = extract_json_payload(payload)
            RubyLLM.logger.debug("Extracted JSON payload: #{json_payload[0..100]}...")
            parse_and_process_json(json_payload, &)
          rescue JSON::ParserError => e
            log_json_parse_error(e, json_payload)
          rescue StandardError => e
            log_general_error(e)
          end

          private

          def extract_json_payload(payload)
            json_start = payload.index('{')
            json_end = payload.rindex('}')
            payload[json_start..json_end]
          end

          def parse_and_process_json(json_payload, &)
            json_data = JSON.parse(json_payload)
            process_json_data(json_data, &)
          end

          def process_json_data(json_data, &)
            return unless json_data['bytes']

            data = decode_and_parse_data(json_data)
            create_and_yield_chunk(data, &)
          end

          def decode_and_parse_data(json_data)
            RubyLLM.logger.debug("Decoding bytes: #{json_data['bytes'][0..30]}...")
            decoded_bytes = Base64.strict_decode64(json_data['bytes'])
            RubyLLM.logger.debug("Decoded bytes (first 100 chars): #{decoded_bytes[0..100]}")
            parsed_data = JSON.parse(decoded_bytes)
            RubyLLM.logger.debug("Parsed data keys: #{parsed_data.keys.join(', ')}")
            parsed_data
          end

          def create_and_yield_chunk(data, &block)
            block.call(build_chunk(data))
          end

          def build_chunk(data)
            Chunk.new(
              **extract_chunk_attributes(data)
            )
          end

          def extract_chunk_attributes(data)
            {
              role: :assistant,
              model_id: extract_model_id(data),
              content: extract_streaming_content(data),
              input_tokens: extract_input_tokens(data),
              output_tokens: extract_output_tokens(data),
              tool_calls: extract_tool_calls(data)
            }
          end

          def log_json_parse_error(error, json_payload)
            RubyLLM.logger.debug "Failed to parse payload as JSON: #{error.message}"
            RubyLLM.logger.debug "Attempted JSON payload: #{json_payload.inspect}"
          end

          def log_general_error(error)
            RubyLLM.logger.debug "Error processing payload: #{error.message}"
          end
        end
      end
    end
  end
end
