import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import ToastContainer from './ToastContainer';

describe('ToastContainer accessibility', () => {
  it('exposes a polite live region so screen readers announce toasts', () => {
    render(
      <ToastContainer toasts={[{ id: 1, type: 'success', message: 'Forecast ready' }]} />
    );
    const region = screen.getByRole('status');
    expect(region).toHaveAttribute('aria-live', 'polite');
    expect(screen.getByText('Forecast ready')).toBeInTheDocument();
  });
});
