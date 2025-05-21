# frozen_string_literal: true

module RubyLLM
  module Providers
    module Bedrock
      # Models methods for the AWS Bedrock API implementation
      module Models
        # List available models from AWS Bedrock
        # @param connection [RubyLLM::Connection] The connection object
        # @return [Array<RubyLLM::ModelInfo>] List of available models
        def list_models(connection:)
          client = bedrock_client(connection.config)
          
          begin
            # Get the list of foundation models from Bedrock
            response = client.list_foundation_models
            
            # Extract and parse model data
            parse_list_models_response(response, slug, capabilities)
          rescue Aws::Errors::ServiceError, Aws::Errors::NonHttpError => e
            handle_aws_error(e)
          end
        end

        module_function

        # Parse the response from list_foundation_models
        # @param response [Aws::Bedrock::Types::ListFoundationModelsResponse] The response from AWS
        # @param slug [String] The provider slug
        # @param capabilities [Module] The capabilities module
        # @return [Array<RubyLLM::ModelInfo>] List of parsed model information
        def parse_list_models_response(response, slug, capabilities)
          models = Array(response.model_summaries)

          # Filter to include only models we care about
          models.select { |m| m.model_id.include?('claude') }.map do |model_data|
            model_id = model_data.model_id

            ModelInfo.new(
              id: model_id_with_region(model_id, model_data),
              name: model_data.model_name || capabilities.format_display_name(model_id),
              provider: slug,
              family: capabilities.model_family(model_id),
              created_at: nil,
              context_window: capabilities.context_window_for(model_id),
              max_output_tokens: capabilities.max_tokens_for(model_id),
              modalities: capabilities.modalities_for(model_id),
              capabilities: capabilities.capabilities_for(model_id),
              pricing: capabilities.pricing_for(model_id),
              metadata: extract_model_metadata(model_data)
            )
          end
        end

        # Extract metadata from model data
        # @param model_data [Aws::Bedrock::Types::FoundationModelSummary] Model data from AWS
        # @return [Hash] Extracted metadata
        def extract_model_metadata(model_data)
          {
            provider_name: model_data.provider_name,
            inference_types: model_data.inference_types_supported || [],
            streaming_supported: model_data.response_streaming_supported || false,
            input_modalities: model_data.input_modalities || [],
            output_modalities: model_data.output_modalities || []
          }
        end

        # Test-friendly method that only sets the ID
        # @param model_data [Aws::Bedrock::Types::FoundationModelSummary] Model data from AWS
        # @param slug [String] The provider slug
        # @param _capabilities [Module] The capabilities module (unused)
        # @return [RubyLLM::ModelInfo] Basic model info for testing
        def create_model_info(model_data, slug, _capabilities)
          model_id = model_data.model_id

          ModelInfo.new(
            id: model_id_with_region(model_id, model_data),
            name: model_data.model_name || model_id,
            provider: slug,
            family: 'claude',
            created_at: nil,
            context_window: 200_000,
            max_output_tokens: 4096,
            modalities: { input: ['text'], output: ['text'] },
            capabilities: [],
            pricing: {},
            metadata: {}
          )
        end

        # Generate the correct model ID with region prefix if needed
        # @param model_id [String] The base model ID
        # @param model_data [Aws::Bedrock::Types::FoundationModelSummary] Model data from AWS
        # @return [String] Model ID with region prefix if needed
        def model_id_with_region(model_id, model_data)
          return model_id unless model_data.inference_types_supported&.include?('INFERENCE_PROFILE')
          return model_id if model_data.inference_types_supported&.include?('ON_DEMAND')

          "us.#{model_id}"
        end
      end
    end
  end
end