import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Navbar from './Navbar';

describe('Navbar', () => {
  it('renders the clear-models control when the handler is provided', () => {
    const onClear = vi.fn();
    render(<Navbar theme="dark" onToggleTheme={() => {}} onClearBrowserModels={onClear} />);
    const btn = screen.getByRole('button', { name: /clear locally trained browser models/i });
    fireEvent.click(btn);
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it('omits the clear-models control when no handler is provided', () => {
    render(<Navbar theme="dark" onToggleTheme={() => {}} />);
    expect(
      screen.queryByRole('button', { name: /clear locally trained browser models/i })
    ).not.toBeInTheDocument();
  });

  it('theme toggle remains accessible', () => {
    const onToggle = vi.fn();
    render(<Navbar theme="light" onToggleTheme={onToggle} />);
    fireEvent.click(screen.getByRole('button', { name: 'Toggle theme' }));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });
});
