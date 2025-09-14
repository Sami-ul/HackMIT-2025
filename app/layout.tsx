import './globals.css'

export const metadata = {
  title: 'Startup Map',
  description: 'Interactive map of YCombinator startups',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}