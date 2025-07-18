# MyAssistant Frontend

A modern, ChatGPT-like AI assistant interface built with Next.js, TypeScript, and Tailwind CSS.

## ✨ Features

- 🎨 **Modern Dark/Light Theme** - Beautiful UI with dark mode support
- 💬 **Real-time Chat Interface** - Smooth conversation flow
- 📱 **Responsive Design** - Works perfectly on all devices
- ⚡ **Fast & Optimized** - Built with Next.js 14 and App Router
- 🔄 **Auto-resizing Input** - Smart textarea that grows with content
- 📜 **Message History** - Clean message display with timestamps
- 🎯 **Loading States** - Beautiful loading animations
- 🚨 **Error Handling** - User-friendly error messages
- 🎨 **Custom Scrollbars** - Styled scrollbars for better UX

## 🛠️ Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety and better development experience
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful, customizable icons
- **Inter Font** - Clean, modern typography

## 🚀 Getting Started

### Prerequisites

Make sure you have the backend server running on `http://localhost:8010`

### Installation

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## 🎨 UI Features

### Dark Mode Support
- Automatic dark mode detection
- Custom scrollbars for both themes
- Smooth color transitions

### Chat Interface
- **Initial State**: Centered welcome screen with input
- **Chat State**: Messages with bottom input
- **Message Types**: User (blue, right) and Assistant (gray, left)
- **Loading**: Animated dots while waiting for response
- **Error Handling**: Red-styled error messages

### Responsive Design
- Works on desktop, tablet, and mobile
- Adaptive message bubbles
- Touch-friendly interface

## 📁 Project Structure

```
src/
├── app/
│   ├── layout.tsx      # Root layout with dark mode
│   ├── page.tsx        # Main chat interface
│   └── globals.css     # Global styles and scrollbars
└── ...
```

## 🔧 Configuration Files

- `tailwind.config.js` - Tailwind CSS configuration with dark mode
- `postcss.config.js` - PostCSS configuration
- `tsconfig.json` - TypeScript configuration
- `next.config.js` - Next.js configuration

## 🌐 API Integration

The frontend communicates with the FastAPI backend at `http://localhost:8010`:

- `POST /query` - Send messages to the LLM
- `GET /health` - Health check endpoint

## 📝 Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## 🎯 Key Improvements

- **Better Typography** - Inter font for improved readability
- **Enhanced Animations** - Smooth transitions and loading states
- **Improved Accessibility** - Better focus states and keyboard navigation
- **Custom Scrollbars** - Styled scrollbars for better UX
- **Error Handling** - Comprehensive error states with user feedback
