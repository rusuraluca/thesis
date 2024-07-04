require 'test_helper'

class InquiresControllerTest < ActionDispatch::IntegrationTest
  test "should get new" do
    get inquires_new_url
    assert_response :success
  end

  test "should get create" do
    get inquires_create_url
    assert_response :success
  end

  test "should get show" do
    get inquires_show_url
    assert_response :success
  end

  test "should get index" do
    get inquires_index_url
    assert_response :success
  end

  test "should get edit" do
    get inquires_edit_url
    assert_response :success
  end

  test "should get update" do
    get inquires_update_url
    assert_response :success
  end

end
