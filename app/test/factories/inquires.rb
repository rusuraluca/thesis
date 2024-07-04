FactoryBot.define do
  factory :inquire do
    date_taken { "2024-05-12" }
    city_taken { "MyString" }
    country_taken { "MyString" }
    similarity { 1.5 }
    profile { nil }
    user { nil }
  end
end
