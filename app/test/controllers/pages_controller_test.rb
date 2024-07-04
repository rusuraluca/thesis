require "test_helper"

class PagesControllerTest < ActionDispatch::IntegrationTest
  test "a user can visit the root URL" do
    get(root_url)
    assert_response(:ok)
    assert_template(:home)
  end
end
