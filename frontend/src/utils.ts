export const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024;

export function formatConfidence(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatClassName(value: string): string {
  return value.replace(/___/g, ' · ').replace(/_/g, ' ');
}

export function validatePlantImage(
  file: Pick<File, 'type' | 'size'>,
): string | null {
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    return 'Choose a JPEG, PNG, or WEBP image.';
  }
  if (file.size > MAX_IMAGE_SIZE_BYTES) {
    return 'Image must be 10 MB or smaller.';
  }
  return null;
}
