import { describe, expect, it } from 'vitest';
import {
  formatClassName,
  formatConfidence,
  MAX_IMAGE_SIZE_BYTES,
  validatePlantImage,
} from './utils';

describe('frontend display helpers', () => {
  it('formats confidence as a percentage with one decimal place', () => {
    expect(formatConfidence(0.02366)).toBe('2.4%');
  });

  it('turns model class identifiers into readable labels', () => {
    expect(formatClassName('Apple___Apple_scab')).toBe('Apple · Apple scab');
  });
});

describe('plant image validation', () => {
  it.each(['image/jpeg', 'image/png', 'image/webp'])('accepts %s images up to the limit', (type) => {
    expect(validatePlantImage({ type, size: MAX_IMAGE_SIZE_BYTES })).toBeNull();
  });

  it('rejects unsupported file formats', () => {
    expect(validatePlantImage({ type: 'image/svg+xml', size: 100 })).toMatch(/JPEG, PNG, or WEBP/);
  });

  it('rejects images larger than 10 MB', () => {
    expect(validatePlantImage({ type: 'image/jpeg', size: MAX_IMAGE_SIZE_BYTES + 1 })).toMatch(/10 MB/);
  });
});
