class CreateInquires < ActiveRecord::Migration[6.0]
  def change
    create_table :inquires do |t|
      t.date :date_taken
      t.string :city_taken
      t.string :country_taken
      t.float :similarity
      t.references :profile, null: false, foreign_key: true
      t.references :user, null: false, foreign_key: true

      t.timestamps
    end
  end
end
