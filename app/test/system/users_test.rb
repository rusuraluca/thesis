require "test_helper"
require "application_system_test_case"

class UsersTest < ApplicationSystemTestCase
  test "a user can visit the root URL, click register and create an account" do
    visit(root_url)
    register_buttons = page.find_all("a", text: I18n.t("register"))
    assert_equal(2, register_buttons.count)
    register_buttons.first.click

    assert_selector("h1", text: I18n.t("register"))

    fill_in("Email", with: "someone@example.com")
    fill_in("Password", with: "some-strong-password")
    fill_in("Password confirmation", with: "some-strong-password")

    find('input[name="commit"]').click

    assert(has_text?(I18n.t("devise.registrations.signed_up_but_unconfirmed")))
  end

  test "a user cannot register an account if the email address is taken" do
    user = FactoryBot.create(:user)
    email = user.email

    visit(root_url)
    register_buttons = page.find_all("a", text: I18n.t("register"))
    assert_equal(2, register_buttons.count)
    register_buttons.first.click

    assert_selector("h1", text: I18n.t("register"))

    fill_in("Email", with: email)
    fill_in("Password", with: "some-strong-password")
    fill_in("Password confirmation", with: "some-strong-password")

    find('input[name="commit"]').click

    assert(has_text?("Email has already been taken"))
  end

  test "a user cannot register an account without specifying a password" do
    visit(root_url)
    register_buttons = page.find_all("a", text: I18n.t("register"))
    assert_equal(2, register_buttons.count)
    register_buttons.first.click

    assert_selector("h1", text: I18n.t("register"))

    fill_in("Email", with: "someone@example.com")

    find('input[name="commit"]').click

    assert(has_text?("Password can't be blank"))
  end

  test "a user cannot register an account without a matching password confirmation" do
    visit(root_url)
    register_buttons = page.find_all("a", text: I18n.t("register"))
    assert_equal(2, register_buttons.count)
    register_buttons.first.click

    assert_selector("h1", text: I18n.t("register"))

    fill_in("Email", with: "someone@example.com")
    fill_in("Password", with: "some-strong-password")
    fill_in("Password confirmation", with: "this is not the same password")

    find('input[name="commit"]').click

    assert(has_text?("Password confirmation doesn't match Password"))
  end

  test "a user can login successfully" do
    user = FactoryBot.create(:user)

    visit(root_url)
    login_buttons = page.find_all("a", text: I18n.t("login"))
    assert_equal(2, login_buttons.count)
    login_buttons.first.click

    assert_selector("h1", text: I18n.t("login"))

    fill_in("Email", with: user.email)
    fill_in("Password", with: "some-strong-password")

    find('input[name="commit"]').click

    assert(has_text?(I18n.t("devise.sessions.signed_in")))
  end

  test "a user cannot login if they have not confirmed their email address" do
    user = FactoryBot.create(:user, :unconfirmed)

    visit(root_url)
    login_buttons = page.find_all("a", text: I18n.t("login"))
    assert_equal(2, login_buttons.count)
    login_buttons.first.click

    assert_selector("h1", text: I18n.t("login"))

    fill_in("Email", with: user.email)
    fill_in("Password", with: "some-strong-password")

    find('input[name="commit"]').click

    assert(has_text?(I18n.t("devise.failure.unconfirmed")))
  end

  test "a user cannot login if they provide the incorrect username" do
    FactoryBot.create(:user, :unconfirmed)

    visit(root_url)
    login_buttons = page.find_all("a", text: I18n.t("login"))
    assert_equal(2, login_buttons.count)
    login_buttons.first.click

    assert_selector("h1", text: I18n.t("login"))

    fill_in("Email", with: "thisisthewrongusername@example.com")
    fill_in("Password", with: "some-strong-password")

    find('input[name="commit"]').click

    assert(
      has_text?(
        I18n.t("devise.failure.invalid", authentication_keys: "Email")
      )
    )
  end

  test "a user cannot login if they provide the incorrect password" do
    user = FactoryBot.create(:user, :unconfirmed)

    visit(root_url)
    login_buttons = page.find_all("a", text: I18n.t("login"))
    assert_equal(2, login_buttons.count)
    login_buttons.first.click

    assert_selector("h1", text: I18n.t("login"))

    fill_in("Email", with: user.email)
    fill_in("Password", with: "") # no password

    find('input[name="commit"]').click

    assert(
      has_text?(
        I18n.t("devise.failure.invalid", authentication_keys: "Email")
      )
    )
  end

  test "a user can say that they have forgotten their password" do
    user = FactoryBot.create(:user, :unconfirmed)

    visit(root_url)
    login_buttons = page.find_all("a", text: I18n.t("login"))
    assert_equal(2, login_buttons.count)
    login_buttons.first.click

    assert_selector("h1", text: I18n.t("login"))
    assert(has_text?("Forgot your password?"))
    click_on("Forgot your password?")

    assert_selector("h1", text: "Forgot your password?")
    fill_in("Email", with: user.email)

    click_on("Send me reset password instructions")

    assert(has_text?(I18n.t("devise.passwords.send_paranoid_instructions")))
  end

  test "a user cannot reveal the passwords that are on the system by the forgot
        password mechanism" do
    visit(root_url)
    login_buttons = page.find_all("a", text: I18n.t("login"))
    assert_equal(2, login_buttons.count)
    login_buttons.first.click

    assert_selector("h1", text: I18n.t("login"))
    assert(has_text?("Forgot your password?"))
    click_on("Forgot your password?")

    assert_selector("h1", text: "Forgot your password?")
    fill_in("Email", with: "someone@example.com")

    click_on("Send me reset password instructions")

    assert(has_text?(I18n.t("devise.passwords.send_paranoid_instructions")))
  end

  test "a user can resend confirmation instructions" do
    user = FactoryBot.create(:user)

    visit(root_url)
    login_buttons = page.find_all("a", text: I18n.t("login"))
    assert_equal(2, login_buttons.count)
    login_buttons.first.click

    assert(has_text?("Didn't receive confirmation instructions?"))
    click_on("Didn't receive confirmation instructions?")

    assert_selector("h1", text: "Resend confirmation instructions")
    fill_in("Email", with: user.email)

    click_on("Resend confirmation instructions")

    assert(has_text?(I18n.t("devise.confirmations.send_paranoid_instructions")))
  end

  test "a user can resend unlock instructions" do
    user = FactoryBot.create(:user)

    visit(root_url)
    login_buttons = page.find_all("a", text: I18n.t("login"))
    assert_equal(2, login_buttons.count)
    login_buttons.first.click

    assert(has_text?("Didn't receive unlock instructions?"))
    click_on("Didn't receive unlock instructions?")

    assert_selector("h1", text: "Resend unlock instructions")
    fill_in("Email", with: user.email)

    click_on("Resend unlock instructions")

    assert(has_text?(I18n.t("devise.unlocks.send_paranoid_instructions")))
  end
end
