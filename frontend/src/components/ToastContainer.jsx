import React from 'react';

const TOAST_ICONS = {
  success: '✅',
  error: '❌',
  info: 'ℹ️',
};

export default function ToastContainer({ toasts }) {
  return (
    <div
      id="toastContainer"
      className="toast-container"
      role="status"
      aria-live="polite"
      aria-atomic="false"
    >
      {toasts.map((t) => (
        <div key={t.id} className="toast">
          <span className="toast-icon" aria-hidden="true">{TOAST_ICONS[t.type] || 'ℹ️'}</span>
          <span className="toast-msg">{t.message}</span>
        </div>
      ))}
    </div>
  );
}
