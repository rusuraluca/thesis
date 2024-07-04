class CreateProfiles < ActiveRecord::Migration[6.0]
  def change
    create_table :profiles do |t|
      t.string :name
      t.date :date_of_birth
      t.string :gender
      t.string :nationality
      t.date :date_of_disappearance
      t.string :city_of_disappearance
      t.string :country_of_disappearance
      t.boolean :found

      t.timestamps
    end
  end
end
