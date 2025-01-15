
# Benji ML Model Repository

Welcome to the **Benji ML Model Repository**! This repository contains the code and resources for the AI-Powered Personal Financial Management System, designed to simplify and enhance personal financial management through Artificial Intelligence (AI) and Machine Learning (ML).

## Project Overview

The Benji ML Model leverages advanced technologies to analyze user financial data, predict spending patterns, and provide actionable insights. It integrates seamlessly with APIs and features user-friendly interfaces for a streamlined experience.

### Key Features:
- **AI-Driven Financial Insights**: Analyze user financial behavior to offer personalized advice.
- **Expense Tracking and Categorization**: Monitor and categorize expenses effortlessly.
- **Predictive Analytics**: Forecast future financial trends to support proactive decision-making.
- **User-Friendly Interface**: Simplified navigation with clear visualizations.
- **Educational Modules**: Improve financial literacy with integrated learning tools.

---

## Project Structure

### Key Files and Directories:

- **lib/core/config/api_endpoints.dart**: Defines API endpoints for integration.
- **lib/data/api/auth_service.dart**: Manages API requests for authentication, registration, and logout.
- **lib/controllers/auth_controller.dart**: Handles state management for user authentication.
- **lib/data/repositories/auth_repository.dart**: Acts as a bridge between API services and controllers, ensuring clean architecture.

---

## Requirements

### Dependencies:
This project uses a mix of Flutter for the frontend and Laravel for backend APIs. Key dependencies include:

**Backend:**
- PHP 8.1+
- Laravel Framework 10.10
- MySQL database
- Google APIs (Client Services)
- Laravel Sanctum for authentication

**Frontend:**
- Flutter SDK
- Dart >=3.4.3
- Intl (for internationalization)
- Lottie (animations)
- Provider (state management)

**Development Tools:**
- XAMPP (for local backend server)
- Visual Studio Code (IDE)
- Google Colab (for AI model development)

---

## Environment Configuration

### Backend:
Ensure the `.env` file contains the correct configuration for database and API keys:

```env
APP_NAME=Laravel
APP_ENV=local
DB_CONNECTION=mysql
DB_HOST=localhost
DB_PORT=3306
DB_DATABASE=personalfinancialmanagementsystem
DB_USERNAME=root
DB_PASSWORD=your_password
MAIL_MAILER=smtp
MAIL_HOST=127.0.0.1
MAIL_PORT=1025
```

### Frontend:
Update `pubspec.yaml` for required Flutter dependencies:

```yaml
dependencies:
  flutter:
    sdk: flutter
  intl: ^0.19.0
  provider: ^6.1.2
  lottie: ^2.1.0
  fl_chart: ^0.69.0
```

---

## Getting Started

### Backend Setup:
1. Clone the repository.
2. Navigate to the Laravel backend directory.
3. Install dependencies:
   ```bash
   composer install
   ```
4. Set up the database:
   ```bash
   php artisan migrate
   php artisan db:seed
   ```
5. Start the server:
   ```bash
   php artisan serve
   ```

### Frontend Setup:
1. Clone the Flutter repository.
2. Install dependencies:
   ```bash
   flutter pub get
   ```
3. Run the application:
   ```bash
   flutter run
   ```

---

## Contribution

Contributions are welcome! Please adhere to the following guidelines:
1. Fork the repository and create a new branch for your feature or bugfix.
2. Write clean, well-documented code.
3. Ensure all tests pass before submitting a pull request.
4. Submit a detailed pull request with a description of your changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For further inquiries or support:
- **Name**: K.K.Y. Vidnath
- **Email**: ykandalama@gmail.com

Thank you for contributing to Benji ML Model!
