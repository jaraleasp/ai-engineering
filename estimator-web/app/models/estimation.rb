class Estimation < ApplicationRecord
  validates :description, :project_type, :detail_level, :output_format, presence: true

  def to_response
    EstimationResponse.from_hash(response_payload)
  end

  def description_preview(limit: 80)
    description.to_s.truncate(limit)
  end
end
