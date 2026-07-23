import React from 'react';

const TOAST_ICONS = {
  success: '✅',
  error: '❌',
  info: 'ℹ️',
};

export default function ToastContainer({ toasts }) {
  return (
    <div id="toastContainer" className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className="toast">
          <span className="toast-icon">{TOAST_ICONS[t.type] || 'ℹ️'}</span>
          <span className="toast-msg">{t.message}</span>
        </div>
      ))}
    </div>
  );
}
