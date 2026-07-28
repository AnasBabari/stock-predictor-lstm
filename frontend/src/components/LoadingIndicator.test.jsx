import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import LoadingIndicator from './LoadingIndicator';

it('renders the server-reported loading stage without percentage progress', () => {
  render(<LoadingIndicator isLoading stage="Training a new model for this ticker…" />);

  expect(screen.getByText('Training a new model for this ticker…')).toBeInTheDocument();
  expect(screen.queryByText(/%/)).not.toBeInTheDocument();
});
