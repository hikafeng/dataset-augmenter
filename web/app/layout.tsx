import '@/styles/globals.css'

export const metadata = {
  title: 'Dataset Augmenter',
  description: 'Image preprocessing & augmentation tool',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
