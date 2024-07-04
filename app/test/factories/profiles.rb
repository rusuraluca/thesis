FactoryBot.define do
  factory :profile do
    name { "MyString" }
    date_of_birth { "2024-05-11" }
    gender { "MyString" }
    nationality { "MyString" }
    date_of_disappearance { "2024-05-11" }
    city_of_disappearance { "MyString" }
    country_of_disappearance { "MyString" }
    found { false }
  end
end
