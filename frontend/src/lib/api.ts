import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''
const apiBaseURL = API_BASE_URL ? `${API_BASE_URL.replace(/\/$/, '')}/api` : '/api'

export const api = axios.create({
  baseURL: apiBaseURL,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default api
