module ApplicationHelper
  def user_is_admin?
    # Checks if the current user is an admin.
    # @return [Boolean] true if the user is signed in and has the admin role, false otherwise.
    user_signed_in? && current_user.has_role?(:admin)
  end

  def toast_klass(flash_message_klass)
    # Returns the appropriate CSS class for toast notifications based on the flash message type.
    # @param flash_message_klass [String, Symbol] the type of flash message (:alert, :notice, etc.).
    # @return [String] the CSS class for the toast notification.
    case flash_message_klass.to_sym
    when :alert
      "toast align-items-center bg-danger border-0"
    else
      "toast align-items-center bg-success border-0"
    end
  end
end
