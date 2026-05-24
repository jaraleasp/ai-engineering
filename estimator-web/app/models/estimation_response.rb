class EstimationResponse
  include ActiveModel::Model
  include ActiveModel::Attributes

  attribute :result               # EstimationResult instance
  attribute :prompt_version, :string
  attribute :cached, :boolean, default: false

  def self.from_hash(hash)
    new(
      result: EstimationResult.new(hash["result"].to_h),
      prompt_version: hash["prompt_version"],
      cached: hash["cached"] || false
    )
  end
end
