# frozen_string_literal: true

require 'aws-sdk-bedrock'
require 'aws-sdk-bedrockruntime'
require 'json'
require_relative 'bedrock/streaming'
require_relative 'anthropic/tools'

module RubyLLM
  module Providers
    # AWS Bedrock API integration. Handles chat completion and streaming
    # for Claude models using the official AWS SDK.
    module Bedrock
      extend Provider
      extend Bedrock::Chat
      extend Bedrock::Streaming
      extend Bedrock::Models
      extend Bedrock::Media
      extend Anthropic::Tools

      # Standard configuration options for all Bedrock requests
      DEFAULTS = {
        anthropic_version: 'bedrock-2023-05-31'
      }.freeze

      # Maximum retry attempts for transient errors
      MAX_RETRIES = 3

      module_function

      def api_base(config)
        "https://bedrock-runtime.#{config.bedrock_region}.amazonaws.com"
      end

      # Parse error messages from API responses
      # @param response [Faraday::Response] The HTTP response
      # @return [String, nil] Parsed error message or nil if not found
      def parse_error(response)
        return if response.body.empty?

        body = try_parse_json(response.body)
        case body
        when Hash
          body['message'] || body.dig('error', 'message')
        when Array
          body.map { |part| part['message'] || part.dig('error', 'message') }.compact.join('. ')
        else
          body
        end
      end

      # Creates and caches a Bedrock client for management operations
      # @param config [RubyLLM::Configuration] The configuration
      # @return [Aws::Bedrock::Client] Configured AWS Bedrock client
      def bedrock_client(config)
        @bedrock_client ||= {}
        @bedrock_client[config] ||= begin
          client_options = {
            region: config.bedrock_region,
            credentials: create_credentials(config),
            retry_limit: MAX_RETRIES,
            retry_mode: 'adaptive'
          }

          Aws::Bedrock::Client.new(client_options)
        end
      rescue Aws::Errors::MissingCredentialsError => e
        raise Error::AuthenticationError.new(
          provider: slug,
          message: "AWS credentials missing or invalid: #{e.message}"
        )
      rescue Aws::Errors::ServiceError => e
        raise Error::ServiceError.new(
          provider: slug,
          message: "AWS Bedrock service error: #{e.message}",
          status: e.context&.http_response&.status_code
        )
      end

      # Creates and caches a BedrockRuntime client for model invocation
      # @param config [RubyLLM::Configuration] The configuration
      # @return [Aws::BedrockRuntime::Client] Configured AWS Bedrock Runtime client
      def bedrock_runtime_client(config)
        @bedrock_runtime_client ||= {}
        @bedrock_runtime_client[config] ||= begin
          client_options = {
            region: config.bedrock_region,
            credentials: create_credentials(config),
            retry_limit: MAX_RETRIES,
            retry_mode: 'adaptive'
          }

          Aws::BedrockRuntime::Client.new(client_options)
        end
      rescue Aws::Errors::MissingCredentialsError => e
        raise Error::AuthenticationError.new(
          provider: slug,
          message: "AWS credentials missing or invalid: #{e.message}"
        )
      rescue Aws::Errors::ServiceError => e
        raise Error::ServiceError.new(
          provider: slug,
          message: "AWS Bedrock Runtime service error: #{e.message}",
          status: e.context&.http_response&.status_code
        )
      end

      # Create AWS credentials from RubyLLM config
      # @param config [RubyLLM::Configuration] The configuration
      # @return [Aws::Credentials] AWS credentials object
      def create_credentials(config)
        Aws::Credentials.new(
          config.bedrock_api_key,
          config.bedrock_secret_key,
          config.bedrock_session_token
        )
      end

      # Handle AWS-specific errors and convert them to RubyLLM errors
      # @param error [Exception] The original error
      # @raise [RubyLLM::Error] Appropriate RubyLLM error based on the AWS error
      def handle_aws_error(error)
        case error
        when Aws::Errors::MissingCredentialsError
          raise Error::AuthenticationError.new(
            provider: slug,
            message: "AWS credentials missing or invalid: #{error.message}"
          )
        when Aws::Errors::ServiceError
          # Map certain error types to specific RubyLLM errors
          case error.class.to_s
          when 'Aws::BedrockRuntime::Errors::ValidationException'
            # Fallback to standard error if ValidationError is not defined
            if defined?(Error::ValidationError)
              raise Error::ValidationError.new(
                provider: slug,
                message: "AWS Bedrock validation error: #{error.message}",
                status: error.context&.http_response&.status_code
              )
            else
              raise Error::ServiceError.new(
                provider: slug,
                message: "AWS Bedrock validation error: #{error.message}",
                status: error.context&.http_response&.status_code
              )
            end
          when 'Aws::BedrockRuntime::Errors::AccessDeniedException'
            raise Error::AuthenticationError.new(
              provider: slug,
              message: "AWS Bedrock access denied: #{error.message}",
              status: error.context&.http_response&.status_code
            )
          when 'Aws::BedrockRuntime::Errors::ModelNotReadyException'
            raise Error::UnavailableError.new(
              provider: slug,
              message: "AWS Bedrock model not ready: #{error.message}",
              status: error.context&.http_response&.status_code
            )
          when 'Aws::BedrockRuntime::Errors::ThrottlingException'
            raise Error::RateLimitError.new(
              provider: slug,
              message: "AWS Bedrock rate limit exceeded: #{error.message}",
              status: error.context&.http_response&.status_code
            )
          when 'Aws::BedrockRuntime::Errors::ModelStreamErrorException'
            raise Error::StreamingError.new(
              provider: slug,
              message: "AWS Bedrock streaming error: #{error.message}",
              status: error.context&.http_response&.status_code
            )
          else
            # Generic service error for other cases
            raise Error::ServiceError.new(
              provider: slug,
              message: "AWS Bedrock service error: #{error.message}",
              status: error.context&.http_response&.status_code
            )
          end
        when Aws::Errors::NonHttpError
          raise Error::ConnectionError.new(
            provider: slug,
            message: "AWS Bedrock connection error: #{error.message}"
          )
        else
          # Re-raise any other errors
          raise error
        end
      end

      def capabilities
        Bedrock::Capabilities
      end

      def slug
        'bedrock'
      end

      def configuration_requirements
        %i[bedrock_api_key bedrock_secret_key bedrock_region]
      end
    end
  end
end