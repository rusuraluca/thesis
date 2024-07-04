require "test_helper"
require "application_system_test_case"

class PagesTest < ApplicationSystemTestCase
  test "a user can visit the root URL and see the app's name" do
    visit(root_url)
    assert_selector("h1", text: I18n.t("app_name"))
  end

  test "a user can visit the root URL and see the app's description" do
    visit(root_url)
    assert_selector("p", text: I18n.t("app_description"))
  end

  test "a user can visit the root URL and see a link to GitHub" do
    visit(root_url)
    assert_selector("a", text: I18n.t("github"))
  end

  test "a user can visit the root URL and see a link to login" do
    visit(root_url)
    assert_selector("a", text: I18n.t("login"))
  end

  test "a user can visit the root URL and see a link to register" do
    visit(root_url)
    assert_selector("a", text: I18n.t("register"))
  end
end
