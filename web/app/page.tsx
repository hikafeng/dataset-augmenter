export default function Home() {
  return (
    <main className="p-8">
      <h1 className="text-2xl font-bold mb-4">Projects</h1>
      <ul className="space-y-2">
        <li className="p-4 border rounded hover:bg-gray-50">
          <a href="/project/123/version/new" className="text-indigo-600 hover:underline">
            Sample Project
          </a>
        </li>
      </ul>
    </main>
  )
}
