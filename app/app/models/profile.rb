class Profile < ApplicationRecord
  # Represents a user profile, which can have multiple inquiries and attached images.
  has_many_attached :images
  has_many :inquires, dependent: :destroy

  def age
    # Calculates the age of the profile owner based on the date_of_birth.
    # @return [Integer, nil] the age in years or nil if date_of_birth is not present.
    ((Time.zone.now - date_of_birth.to_time) / 1.year.seconds).floor if date_of_birth
  end
end
