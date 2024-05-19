import './movie.css';
import Image from 'next/image';
import { useSession } from 'next-auth/react';
import { getServerSession } from 'next-auth';
import { authOptions } from '~/server/auth';
import { useRouter } from 'next/navigation';
import Navbar from '../../_components/navbar/navbar';
import { api } from '~/trpc/server';
import { TextField } from '@mui/material';
import { Movie } from '~/server/db/schema';
import StarRating from '../../_components/star-rating';



export default async function MoviePage({params}) {
  const session = await getServerSession(authOptions);
  console.log("movie params", params);
  const movieId = Number(params['id']);

  if (!session) {
    redirect('/login');
  }




  const movie: Movie = await api.movie.byId(movieId)

  return (
    <>
      <Navbar></Navbar>
      <div className='movie-container'>
        <h1>{movie.title}</h1>
        <p>{movie.genres}</p>
        <StarRating />
 
        <TextField
          id="outlined-textarea"
          label="Review"
          multiline
          rows={10}
          defaultValue=""
          variant="filled"
        />
      </div>
    </>
  );
}


// props: 
// movie to review
// logic
// form handling for review
// update db with review