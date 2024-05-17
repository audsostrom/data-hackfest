import './home.css';
import Image from 'next/image';
import Rows from '../../../public/images/row.jpeg';
import Profile from '../../../public/images/profile.svg';
export default function Home() {
  return (
    <div className='home-container'>
      {

        /**
               <h1 >Home Page</h1>
      <p style={{ fontFamily: 'Gurajada', fontSize: 70 }}>Welcome to the Home page!</p>
      <div style={{ fontFamily: 'DMSans' }}>Testing font imports</div>
         */
      }
      <div className='left-side'>
        <div className='profile-section'>
          <div className="banner">
          </div>
          <Image className='logo' src={Profile} alt={'profile'} width='60' height='60'></Image>
          <div>Audrey Ostrom</div>
          <div>@audstrom</div>
          <div className='profile-stats'>
            <div className='movies-seen'>
              <div>18</div>
              <div>Movies Seen</div>
            </div>
            <div className='friends-num'>
              <div>18</div>
              <div>Friends</div>
            </div>
            <div className='bucketlists-num'>
              <div>18</div>
              <div>Bucket Lists</div>
            </div>

          </div>
        </div>
        <div className='friend-activity'>
          <div>Friend Activity</div>
          <ol>
            <li>Riley watched to Kill a MockingBird</li>
            <li>Riley watched to Kill a MockingBird</li>
            <li>Riley watched to Kill a MockingBird</li>
            <li>Riley watched to Kill a MockingBird</li>
          </ol>
        </div>
      </div>
      <div className='middle-options'>
        <div>Find Your Next Movie</div>
        <div>Are you watching by yourself? &#xf059;</div>
        <div>Pick for me! &#xf059;</div>
        <div>Take a quiz &#xf059;</div>
        <div>Are you watching with a group? &#xf059;</div>
        <div>Pick for us! &#xf059;</div>
        <div>Work together on a quiz &#xf059;</div>
      </div>
      <div className='right-side'>
        <div>All About You</div>
        <div>You're subscribed to</div>
        <div>
          <div>Netflix</div>
          <div>Netflix</div>
          <div>Netflix</div>
          <div>Netflix</div>
        </div>
        <div>You love these genres</div>
        <div>Scary, come</div>

      </div>

    </div>
  );
}


