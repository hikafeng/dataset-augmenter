// filepath: \web\components\Sidebar.tsx
export default function Sidebar() {
  return (
    <aside className="w-64 bg-white border-r p-4 shadow-md rounded">
      <h2 className="text-xl font-bold mb-4">Versions</h2>
      <ul className="space-y-2">
        <li className="text-gray-700">
          <div className="flex justify-between">
            <span>v2</span>
            <span className="text-sm text-gray-500">640×640</span>
          </div>
          <div className="text-sm text-gray-400">2025-07-02 16:37</div>
        </li>
      </ul>
    </aside>
  )
}