# frozen_string_literal: true

module RubyLLM
  module Providers
    module Bedrock
      # Media handling methods for the Bedrock API integration
      module Media
        extend Anthropic::Media

        module_function

        # Format content for the Bedrock API (images, PDFs, etc.)
        # @param content [String, RubyLLM::Content] The content to format
        # @return [Array<Hash>, String] Formatted content for the API
        def format_content(content)
          return [Anthropic::Media.format_text(content)] unless content.is_a?(Content)

          parts = []
          parts << Anthropic::Media.format_text(content.text) if content.text

          content.attachments.each do |attachment|
            case attachment
            when Attachments::Image
              parts << format_image(attachment)
            when Attachments::PDF
              parts << format_pdf(attachment)
            when Attachments::Audio
              raise Error::UnsupportedFeatureError.new(
                provider: 'bedrock',
                message: "Audio attachments are not currently supported with Bedrock"
              )
            else
              raise Error::UnsupportedFeatureError.new(
                provider: 'bedrock',
                message: "Unsupported attachment type: #{attachment.class}"
              )
            end
          end

          parts
        end

        # Format an image for the Bedrock API
        # @param image [RubyLLM::Attachments::Image] The image to format
        # @return [Hash] Formatted image data for the API
        def format_image(image)
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: image.mime_type,
              data: image.encoded
            }
          }
        end

        # Format a PDF for the Bedrock API
        # @param pdf [RubyLLM::Attachments::PDF] The PDF to format
        # @return [Hash] Formatted PDF data for the API
        def format_pdf(pdf)
          {
            type: 'document',
            source: {
              type: 'base64',
              media_type: pdf.mime_type,
              data: pdf.encoded
            }
          }
        end
      end
    end
  end
end